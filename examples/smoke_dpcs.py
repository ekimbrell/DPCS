import argparse, math, time, contextlib
import torch
import torch.nn as nn
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import torch.optim as optim

# ---- Optional: Transformer Engine (FP8) ----
try:
    import transformer_engine.pytorch as te
    from transformer_engine.common import recipe as te_recipe
    TE_AVAILABLE = True
except Exception:
    TE_AVAILABLE = False

# ---- Your scheduler (matches DPCS.py API) ----
from dpcs import DPCS   # ctor: DPCS(**kwargs); methods: wrap(), start_step(), collect_signals(loss, model), end_step(opt, scaler=None)

def vram_headroom_frac(device):
    if device.type != "cuda":
        return None
    # Official API: returns (free_bytes, total_bytes).
    free_b, total_b = torch.cuda.memory.mem_get_info()  # docs. :contentReference[oaicite:2]{index=2}
    return free_b / total_b if total_b else None

# Tiny CNN (no torchvision dep)
class TinyCNN(nn.Module):
    def __init__(self, in_ch=3, classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, classes),
        )
    def forward(self, x): return self.net(x)

@contextlib.contextmanager
def capture_checkpoint_calls():
    import torch.utils.checkpoint as tuc
    import DPCS as dpcs_mod  # to patch the alias imported inside your file

    calls = []   # list of dicts: {"step": int, "use_reentrant": value, "det_check": value}
    step_ref = {"idx": -1}

    def wrapper(function, *args, **kwargs):
        calls.append({
            "step": step_ref["idx"],
            "use_reentrant": kwargs.get("use_reentrant", None),
            "det_check": kwargs.get("determinism_check", None),
        })
        return orig_tuc(function, *args, **kwargs)

    # save originals and patch both entry points
    orig_tuc = tuc.checkpoint
    orig_local = getattr(dpcs_mod, "checkpoint", None)
    tuc.checkpoint = wrapper
    if orig_local is not None:
        setattr(dpcs_mod, "checkpoint", wrapper)

    try:
        yield calls, step_ref
    finally:
        tuc.checkpoint = orig_tuc
        if orig_local is not None:
            setattr(dpcs_mod, "checkpoint", orig_local)


    def wrapper(function, *args, **kwargs):
        calls.append({
            "step": step_ref["idx"],
            "use_reentrant": kwargs.get("use_reentrant", None),
            "det_check": kwargs.get("determinism_check", None),
        })
        return orig(function, *args, **kwargs)

    tuc.checkpoint = wrapper
    try:
        yield calls, step_ref
    finally:
        tuc.checkpoint = orig

# minimal telemetry
class Telemetry:
    def __init__(self):
        self.mix_by_step = []
    def __call__(self, rec):
        if isinstance(rec, dict) and "mix" in rec:
            self.mix_by_step.append((rec.get("step"), rec["mix"]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(0)

    # synthetic data
    x = torch.randn(args.batch, 3, 32, 32, device=device)
    y = torch.randint(0, 10, (args.batch,), device=device)

    model = TinyCNN().to(device)
    opt = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    crit = nn.CrossEntropyLoss()

    # DPCS: construct via kwargs, wrap() the model, attach logger
    tel = Telemetry()
    dpcs = DPCS(
        # hysteresis bands (visible changes in a short run)
        epsilon_g_low=5e-4, epsilon_g_high=2e-3,
        kappa_low=2.5, kappa_high=10.0,
        mode_patience=2,
        # curvature probe (frequent but cheap)
        curv_period=4, hvp_power_iters=3, max_modules_per_probe=2,
        # checkpoint hysteresis (loose to trigger)
        low_headroom_frac=0.60, hi_headroom_frac=0.70, ckpt_patience=1,
        # make small layers eligible for checkpoint in this smoke test
        min_activation_bytes_to_ckpt=0,     # default is 16 MiB; too high for TinyCNN
        ckpt_enable_topk=False,             # bypass top-K gating for this test
        ckpt_min_candidates=1,
        # TE recipe tuning (used only if TE is installed)
        te_amax_history_len=128, te_margin_init=0, te_margin_inc=1, te_margin_dec=1,
        # keep default determinism_check="none" for speed in smoke test
    )
    dpcs.on_log(tel)
    model = dpcs.wrap(model)   # instrumentation
    dpcs.enable_checkpointing(True)

    if TE_AVAILABLE:
        te.common.set_fp8_recipe(te_recipe.DelayedScaling(margin=0, amax_history_len=128))
        print("[TE] FP8 DelayedScaling set (margin=0, amax_history_len=128).")  # API docs. :contentReference[oaicite:3]{index=3}
    else:
        print("[TE] Transformer Engine not available — FP8 paths should no-op.")

    # dtype snapshot hooks (optional peek at precision effects)
    watched = []
    dtypes_by_step = {}
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)) and len(watched) < 2:
            def mk(name_):
                def hook(mod, inp, out):
                    step = getattr(hook_ctx, "step", -1)
                    if isinstance(out, torch.Tensor):
                        dt = out.dtype
                    elif isinstance(out, (tuple, list)) and out and isinstance(out[0], torch.Tensor):
                        dt = out[0].dtype
                    else:
                        dt = None
                    dtypes_by_step.setdefault(step, {})[name_] = dt
                return hook
            watched.append(m.register_forward_hook(mk(name)))

    # reserve buffer to create/release memory pressure
    reserve = None

    # training loop
    with capture_checkpoint_calls() as (ck_calls, step_ref):
        for step in range(args.steps):
            hook_ctx.step = step
            step_ref["idx"] = step

            # induce memory pressure to force checkpointing ON
            if device.type == "cuda":
                if step == 6:
                    free_b, _ = torch.cuda.memory.mem_get_info()
                    target_elems = int(0.60 * free_b // 4)  # float32
                    if target_elems > 0:
                        reserve = torch.empty(target_elems, dtype=torch.float32, device=device)
                        torch.cuda.synchronize()
                        print(f"[mem] Pressure ON at step {step}.")
                if step == 12 and reserve is not None:
                    del reserve
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    print(f"[mem] Pressure OFF at step {step}.")

            # one step
            if device.type == "cuda":
                free_b, total_b = torch.cuda.memory.mem_get_info()  # official API
                print(f"[mem] step {step}: headroom={free_b/total_b:.3f}")
            dpcs.start_step()
            opt.zero_grad(set_to_none=True)
            out = model(x)
            loss = crit(out, y)
            # run curvature probe + stats safely: keep graph for HVP, and have grads for variance
            loss.backward(retain_graph=True)
            dpcs.collect_signals(loss, model)   # uses both p.grad and autograd.grad
            opt.step()
            dpcs.end_step(opt, scaler=None)     # your API

    # cleanup hooks
    for h in watched: h.remove()

    # --- report ---
    print("\n=== SUMMARY ===")
    # checkpoint path flags (non-reentrant recommended & requires explicit flag) :contentReference[oaicite:4]{index=4}
    if ck_calls:
        flags = {c["use_reentrant"] for c in ck_calls}
        dets  = {c["det_check"] for c in ck_calls}
        print(f"[checkpoint] use_reentrant flags seen: {sorted(flags)}")
        print(f"[checkpoint] determinism_check values: {sorted(dets)}")
        if None in flags:
            print("[WARN] Some calls omitted 'use_reentrant'; pass it explicitly. :contentReference[oaicite:5]{index=5}")
        if False in flags:
            print("[OK] Non-reentrant checkpoint detected (use_reentrant=False).")
    else:
        print("[checkpoint] No checkpoint() calls observed.")

    # hysteresis / mode mix snapshots
    if tel.mix_by_step:
        print("[precision] mode mix by step (subset):")
        for s, mix in tel.mix_by_step[:8]:
            print(f"  step {s}: {mix}")

    # dtype samples (fallback view of precision effects)
    if dtypes_by_step:
        some = list(sorted(dtypes_by_step.items()))[:10]
        for s, layer2dt in some:
            lab = ", ".join(f"{k.split('.')[-1]}={str(v).split('.')[-1] if v else None}" for k,v in layer2dt.items())
            print(f"[dtype] step {s}: {lab}")

    if TE_AVAILABLE:
        print("[TE] DelayedScaling was applied; your DPCS will adapt margin when in fp8.")
    else:
        print("[TE] Not installed; FP8 paths are no-ops.")

class HC: pass
hook_ctx = HC()

if __name__ == "__main__":
    raise SystemExit(main())
