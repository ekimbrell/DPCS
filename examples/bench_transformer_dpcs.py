# Benchmark a Transformer and auto-fit batch/seq robustly on small VRAM (Windows-friendly)

import os, gc, time, platform
from collections import Counter
import torch
import torch.nn as nn
from dpcs import DPCS

torch.backends.cudnn.benchmark = False  # avoid long autotune stalls
torch.cuda.memory.set_per_process_memory_fraction(0.70)  # ~70% of visible VRAM
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --- Model dials (kept large; autofit will shrink batch/seq as needed) ---
D_MODEL  = 512
N_HEADS  = 8
FF_DIM   = 2048
N_LAYERS = 8
DROPOUT  = 0.0

# --- Workload starting guesses (autofit refines these) ---
BATCH  = 2
SEQ_LEN = 1024
STEPS  = 60
WARMUP = 20

def mb(x): return int(x / (1024**2)) if x else 0

class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=N_HEADS, dim_feedforward=FF_DIM,
            dropout=DROPOUT, batch_first=True
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=N_LAYERS)
        self.out = nn.Linear(D_MODEL, D_MODEL)

    

    def forward(self, x):
        h = self.enc(x)              # (B, S, D)
        y = self.out(h).mean((1,2))  # (B,) -> scalar later
        return y

def rand_batch(b, s):
    return torch.randn(b, s, D_MODEL, device=DEVICE)

# ---------- OOM-safe probe & autofit ----------
def _safe_cuda_cleanup():
    if DEVICE == "cuda":
        try: torch.cuda.synchronize()
        except Exception: pass
        try: torch.cuda.reset_peak_memory_stats()
        except Exception: pass
        try: torch.cuda.empty_cache()
        except Exception: pass
    gc.collect()

def _is_oom(e: BaseException) -> bool:
    txt = str(e).lower()
    return ("out of memory" in txt) or isinstance(e, getattr(torch.cuda, "OutOfMemoryError", RuntimeError))

def multi_step_fits(b, s, steps=4, warmup=1) -> bool:
    """Require several AMP steps to pass to avoid OOM later."""
    _safe_cuda_cleanup()
    model = TinyTransformer().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda") if DEVICE == "cuda" else None
    x = rand_batch(b, s)
    try:
        for i in range(steps):
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=DEVICE, dtype=(torch.float16 if DEVICE=="cuda" else torch.bfloat16), enabled=True):
                y = model(x); loss = (y**2).mean()
            if scaler:
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else:
                loss.backward(); opt.step()
            if DEVICE == "cuda":
                torch.cuda.synchronize()
        return True
    except Exception as e:
        if _is_oom(e): return False
        raise
    finally:
        del model, opt
        for name in ("x","y","loss"):
            if name in locals(): del locals()[name]
        _safe_cuda_cleanup()

def autofit_workload():
    """Halve batch, then seq, until several steps fit. Apply a small safety margin."""
    global BATCH, SEQ_LEN
    b, s = BATCH, SEQ_LEN
    MIN_SEQ = 128

    while b >= 1 and s >= MIN_SEQ:
        if multi_step_fits(b, s, steps=5, warmup=1):
            # safety margin: shave ~10% off seq and round to a multiple of 64
            s_safe = max(MIN_SEQ, (int(s * 0.9) // 64) * 64)
            print(f"[autofit] found fit at batch={b}, seq={s} -> using batch={b}, seq={s_safe}")
            BATCH, SEQ_LEN = b, s_safe
            return
        if b > 1:
            b = max(b // 2, 1)
        else:
            s //= 2
    raise RuntimeError("Autofit failed. Try lowering D_MODEL / N_LAYERS / FF_DIM.")

# ---------- timing ----------
def measure(step_fn, steps=STEPS, warmup=WARMUP):
    times_ms, max_peak, last_loss = [], 0, 0.0
    for i in range(steps):
        if DEVICE == "cuda":
            torch.cuda.reset_peak_memory_stats()
            start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
            start.record()
            loss = step_fn(i)
            end.record()
            torch.cuda.synchronize()
            if i >= warmup:
                times_ms.append(start.elapsed_time(end))  # ms
                max_peak = max(max_peak, torch.cuda.max_memory_allocated())
        else:
            t0 = time.perf_counter(); loss = step_fn(i)
            if i >= warmup: times_ms.append((time.perf_counter() - t0) * 1000.0)
        last_loss = float(loss)
    avg_ms = sum(times_ms) / max(len(times_ms), 1)
    ips = (BATCH * SEQ_LEN) / (avg_ms / 1000.0) if avg_ms > 0 else 0.0
    return avg_ms, ips, max_peak, last_loss

# ---------------- Baseline AMP ----------------
def bench_baseline():
    model = TinyTransformer().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda") if DEVICE == "cuda" else None
    x = rand_batch(BATCH, SEQ_LEN)
    
    def step(_i):
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=DEVICE, dtype=(torch.float16 if DEVICE=="cuda" else torch.bfloat16), enabled=True):
            y = model(x); loss = (y**2).mean()
        if scaler: scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        else:      loss.backward(); opt.step()
        return loss.detach()
    return measure(step)

# -------------- DPCS adaptive (ckpt gate + precision) --------------
def bench_dpcs_adaptive():
    model = TinyTransformer().to(DEVICE)
    dpcs = DPCS(device_type=DEVICE, epsilon_g=1e-3, kappa=5.0,
                ckpt_low=0.12, ckpt_high=0.20, ckpt_need=2,
                fp8_backend="te")  # FP8 only if TE + supported GPU
    model = dpcs.wrap(model, allow_fp8=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda") if DEVICE == "cuda" else None
    x = rand_batch(BATCH, SEQ_LEN)
    def step(i):
        dpcs.start_step(); opt.zero_grad(set_to_none=True)
        fwd_ctx, _ = dpcs.checkpoint_contexts_if_needed()
        
        torch.cuda.reset_peak_memory_stats()
        with fwd_ctx, torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            y = model(x); loss = (y**2).mean()
        torch.cuda.synchronize()
        fwd_peak_mb = int(torch.cuda.max_memory_allocated()/1024/1024)
        print("forward-only peak MB:", fwd_peak_mb)
        if scaler: scaler.scale(loss).backward(); dpcs.collect_signals(loss, model); scaler.step(opt); scaler.update()
        else:      loss.backward(); dpcs.collect_signals(loss, model); opt.step()
        dpcs.end_step(opt, scaler)
        if DEVICE == "cuda" and (i % 10 == 0):
            free, total = torch.cuda.memory.mem_get_info()
            modes = Counter(st.mode for st in dpcs._registry.values())
            print(f"[step {i:02d}] headroom={free/total:.3f} ckpt_on={dpcs.is_checkpointing()} modes={dict(modes)}")
        return loss.detach()
    avg_ms, ips, max_peak, last_loss = measure(step)
    modes = Counter(st.mode for st in dpcs._registry.values())
    return avg_ms, ips, max_peak, last_loss, dict(modes)

def main():
    print(f"Device: {DEVICE}, CUDA available: {torch.cuda.is_available()}")
    if DEVICE == "cuda":
        free, total = torch.cuda.memory.mem_get_info()
        print(f"GPU Free/Total: {mb(free)}MB / {mb(total)}MB")

    # Auto-fit batch/seq robustly
    autofit_workload()

    b_ms, b_ips, b_peak, b_loss = bench_baseline()
    d_ms, d_ips, d_peak, d_loss, d_modes = bench_dpcs_adaptive()

    print("\n=== Transformer throughput (avg steady-state) ===")
    print(f"Baseline AMP:   {b_ms:.2f} ms/step, {b_ips:.1f} tok/s, peak {mb(b_peak)} MB, last loss {b_loss:.4f}")
    print(f"DPCS adaptive:  {d_ms:.2f} ms/step, {d_ips:.1f} tok/s, peak {mb(d_peak)} MB, last loss {d_loss:.4f}, modes={d_modes}")

if __name__ == '__main__':
    # Avoid allocator warning on Windows; only set allocator config on non-Windows
    if platform.system() != "Windows":
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")
    main()
