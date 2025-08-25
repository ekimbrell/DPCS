# Find "max sequence that fits" for a Transformer:
#  - Baseline AMP
#  - DPCS (adaptive checkpoint gate + precision scheduling)
#
# It auto-falls back to smaller model sizes until it finds a case where
# DPCS fits a longer sequence than baseline. Windows-friendly, with robust
# multi-step OOM probes and CUDA sync/cleanup.

import gc, time, platform
from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import torch.nn as nn
from dpcs import DPCS

torch.backends.cudnn.benchmark = False
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def mb(x): return int(x / (1024**2)) if x else 0

@dataclass
class ModelCfg:
    d_model: int
    n_heads: int
    ff_dim:  int
    n_layers:int
    dropout: float = 0.0

CANDIDATE_CFGS = [ModelCfg(768, 6, 3072, 8)]

BATCH   = 4               # keep batch fixed; we vary sequence length
SEQ_MIN = 128
SEQ_MAX = 4096            # search upper bound
MARGIN  = 64              # require at least +64 tokens win to declare success

# --------- tiny Transformer (parametric) ---------
class TinyTransformer(nn.Module):
    def __init__(self, cfg: ModelCfg):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.ff_dim,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        self.out = nn.Linear(cfg.d_model, cfg.d_model)
        self.d_model = cfg.d_model

    def forward(self, x):
        h = self.enc(x)                 # (B, S, D)
        y = self.out(h).mean((1, 2))    # (B,) -> scalar later
        return y

def rand_batch(b, s, d_model):
    return torch.randn(b, s, d_model, device=DEVICE)

# --------- OOM-safe helpers ----------
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
    t = str(e).lower()
    return ("out of memory" in t) or isinstance(e, getattr(torch.cuda, "OutOfMemoryError", RuntimeError))

# Run a few AMP steps at a given seq; return True if it fits
def multi_step_fits(model: nn.Module, seq: int, dpcs: Optional[DPCS], steps=3, timeout_s=20) -> bool:
    _safe_cuda_cleanup()
    t0 = time.perf_counter()
    try:
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler("cuda") if DEVICE == "cuda" else None
        x = rand_batch(BATCH, seq, model.d_model)

        for i in range(steps):
            if time.perf_counter() - t0 > timeout_s:
                return False  # treat as non-fitting (likely paging or algorithm re-tuning)

            opt.zero_grad(set_to_none=True)
            if dpcs is not None:
                dpcs.start_step()

            fwd_ctx = (dpcs.checkpoint_contexts_if_needed()[0] if dpcs is not None
                       else torch.autocast(device_type=DEVICE, dtype=(torch.float16 if DEVICE=="cuda" else torch.bfloat16), enabled=True))

            with fwd_ctx, torch.autocast(device_type=DEVICE, dtype=(torch.float16 if DEVICE=="cuda" else torch.bfloat16), enabled=True):
                y = model(x); loss = (y**2).mean()

            if scaler:
                scaler.scale(loss).backward()
                if dpcs is not None: dpcs.collect_signals(loss, model)
                scaler.step(opt); scaler.update()
            else:
                loss.backward()
                if dpcs is not None: dpcs.collect_signals(loss, model)
                opt.step()

            if dpcs is not None:
                dpcs.end_step(opt, scaler)

            if DEVICE == "cuda":
                torch.cuda.synchronize()
        return True
    except Exception as e:
        if _is_oom(e): return False
        raise
    finally:
        _safe_cuda_cleanup()


# Binary search max seq that fits for a fixed model (baseline or DPCS path)
def max_seq_for_model(cfg: ModelCfg, use_dpcs: bool) -> int:
    model = TinyTransformer(cfg).to(DEVICE)
    dpcs = None
    if use_dpcs:
        dpcs = DPCS(
            device_type=DEVICE,
            epsilon_g=1e-3, kappa=5.0,
            ckpt_low=0.20, ckpt_high=0.25, ckpt_need=1, # adaptive gate
            fp8_backend="te",                              # FP8 only if TE+GPU supports
            signals_freq_steps=1                           # quicker signal updates for demo
        )
        model = dpcs.wrap(model, allow_fp8=True)

    # Exponential grow to find upper bound
    lo, hi = SEQ_MIN, SEQ_MIN
    while hi <= SEQ_MAX and multi_step_fits(model, hi, dpcs, steps=3):
        lo, hi = hi, hi * 2
        if hi > SEQ_MAX: hi = SEQ_MAX + 1
    if lo < SEQ_MIN:  # even min failed
        return 0

    # Binary search in (lo, hi)
    best = lo
    l, r = lo, min(hi, SEQ_MAX)
    while l + 64 <= r:  # keep 64-token granularity
        mid = ((l + r) // 64) * 64
        ok = multi_step_fits(model, mid, dpcs, steps=3)
        if ok:
            best = mid
            l = mid + 64
        else:
            r = mid - 64
    return best

def run_for_cfg(cfg: ModelCfg) -> Tuple[int, int]:
    base = max_seq_for_model(cfg, use_dpcs=False)
    dpcs = max_seq_for_model(cfg, use_dpcs=True)
    return base, dpcs

def main():
    print(f"Device: {DEVICE}, CUDA available: {torch.cuda.is_available()}")
    if DEVICE == "cuda":
        free, total = torch.cuda.memory.mem_get_info()
        print(f"GPU Free/Total: {mb(free)}MB / {mb(total)}MB")

    found = False
    for i, cfg in enumerate(CANDIDATE_CFGS, 1):
        print(f"\n--- Config {i}: d_model={cfg.d_model}, n_layers={cfg.n_layers}, n_heads={cfg.n_heads}, ff_dim={cfg.ff_dim} ---")
        base_max, dpcs_max = run_for_cfg(cfg)
        print(f"Baseline AMP max seq: {base_max}")
        print(f"DPCS adaptive  max seq: {dpcs_max}")
        if base_max == SEQ_MAX:
            print("(Baseline reached SEQ_MAX; raise SEQ_MAX to continue searching.)")
        if dpcs_max == SEQ_MAX:
            print("(DPCS reached SEQ_MAX; raise SEQ_MAX to continue searching.)")

        if dpcs_max >= base_max + MARGIN:
            print(f"\n✅ DPCS WIN on this config: +{dpcs_max - base_max} tokens at same batch={BATCH}")
            found = True
            break
        else:
            print("…no clear win; trying a smaller model to avoid Windows paging masking effects.")

    if not found:
        print("\nNo clear baseline-vs-DPCS difference found across configs. "
              "On Windows WDDM, VRAM oversubscription can mask memory savings; "
              "try running on Linux/WSL2 or adjust configs for a tighter memory regime.")

if __name__ == "__main__":
    # Avoid allocator tuning warnings on Windows
    if platform.system() != "Windows":
        import os
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")
    main()
