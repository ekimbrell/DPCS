# examples/bench_transformer_dpcs.py
# Benchmark a tiny Transformer with:
#   - Baseline AMP (no checkpoint)
#   - DPCS (no-op)  -> slow signals, no checkpoint
#   - DPCS (ckpt)   -> forced selective activation checkpointing (clear memory win)
#
# Measures step time with CUDA events and prints peak memory + max tokens that fit.
# Requires: torch, and your installed dpcs package (editable install is fine).

import time
from collections import Counter
import gc, os
import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import CheckpointPolicy  # for forcing ckpt in the DPCS case
from dpcs import DPCS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Model size knobs (tune these to push your 4 GB GPU) ---
D_MODEL = 1024         # transformer embed dim
N_HEADS = 8
FF_DIM  = 4096
N_LAYERS = 8
DROPOUT = 0.0          # keep deterministic-ish; not critical for this bench

# --- Workload size knobs ---
BATCH   = 16           # starting batch
SEQ_LEN = 1024         # starting sequence length (tokens)
STEPS   = 60
WARMUP  = 20
SIGNAL_EVERY = 200     # sparse signals → minimal overhead

def mb(x): return int(x / (1024**2)) if x else 0

class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=N_HEADS,
            dim_feedforward=FF_DIM,
            dropout=DROPOUT,
            batch_first=True,  # (B, S, D)
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=N_LAYERS)
        # simple output head so we can form a scalar loss
        self.out = nn.Linear(D_MODEL, D_MODEL)

    def forward(self, x):
        # x: (B, S, D)
        h = self.enc(x)
        y = self.out(h).mean(dim=(1, 2))  # (B,) → scalar loss later
        return y

def rand_batch(batch, seq):
    return torch.randn(batch, seq, D_MODEL, device=DEVICE)

def measure(run_step, steps=STEPS, warmup=WARMUP):
    """Accurate timing: CUDA events on GPU; perf_counter on CPU."""
    times_ms, max_peak, last_loss = [], 0, 0.0
    for i in range(steps):
        if DEVICE == "cuda":
            torch.cuda.reset_peak_memory_stats()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            loss = run_step()
            end.record()
            torch.cuda.synchronize()
            if i >= warmup:
                times_ms.append(start.elapsed_time(end))  # milliseconds
                max_peak = max(max_peak, torch.cuda.max_memory_allocated())
        else:
            t0 = time.perf_counter()
            loss = run_step()
            if i >= warmup:
                times_ms.append((time.perf_counter() - t0) * 1000.0)
        last_loss = float(loss)
    avg_ms = sum(times_ms) / max(len(times_ms), 1)
    tok_per_step = BATCH * SEQ_LEN
    ips = tok_per_step / (avg_ms / 1000.0) if avg_ms > 0 else 0.0
    return avg_ms, ips, max_peak, last_loss

def bench_baseline(batch=BATCH, seq=SEQ_LEN):
    model = TinyTransformer().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda") if DEVICE == "cuda" else None
    x = rand_batch(batch, seq)

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=DEVICE, dtype=(torch.float16 if DEVICE=="cuda" else torch.bfloat16), enabled=True):
            y = model(x)                # (B,)
            loss = (y ** 2).mean()      # scalar
        if scaler:
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        else:
            loss.backward(); opt.step()
        return loss.detach()
    return measure(step)

def bench_dpcs_noop(batch=BATCH, seq=SEQ_LEN):
    model = TinyTransformer().to(DEVICE)
    dpcs = DPCS(device_type=DEVICE, signals_freq_steps=10_000)  # effectively off
    model = dpcs.wrap(model, allow_fp8=(DEVICE=="cuda"))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda") if DEVICE == "cuda" else None
    x = rand_batch(batch, seq)

    def step():
        dpcs.start_step()
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=DEVICE, dtype=(torch.float16 if DEVICE=="cuda" else torch.bfloat16), enabled=True):
            y = model(x)
            loss = (y ** 2).mean()
        if scaler:
            scaler.scale(loss).backward()
            dpcs.collect_signals(loss, model)  # cheap/no-op cadence
            scaler.step(opt); scaler.update()
        else:
            loss.backward(); dpcs.collect_signals(loss, model); opt.step()
        dpcs.end_step(opt, scaler)
        return loss.detach()

    avg_ms, ips, max_peak, last_loss = measure(step)
    modes = Counter(s.mode for s in dpcs._registry.values())
    return avg_ms, ips, max_peak, last_loss, dict(modes)

def bench_dpcs_ckpt(batch=BATCH, seq=SEQ_LEN):
    """Force selective activation checkpointing to demonstrate memory drop."""
    model = TinyTransformer().to(DEVICE)
    dpcs = DPCS(device_type=DEVICE, signals_freq_steps=SIGNAL_EVERY)
    model = dpcs.wrap(model, allow_fp8=(DEVICE=="cuda"))
    # Force the DPCS policy to MUST_RECOMPUTE for the demo (compute→memory trade)
    dpcs._ckpt_policy = CheckpointPolicy.MUST_RECOMPUTE

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda") if DEVICE == "cuda" else None
    x = rand_batch(batch, seq)

    def step(i=[0]):
        dpcs.start_step()
        opt.zero_grad(set_to_none=True)
        # Always enter the selective checkpoint context for this run
        fwd_ctx, _ = dpcs.checkpoint_contexts()
        with fwd_ctx:
            with torch.autocast(device_type=DEVICE, dtype=(torch.float16 if DEVICE=="cuda" else torch.bfloat16), enabled=True):
                y = model(x)
                loss = (y ** 2).mean()
        if scaler:
            scaler.scale(loss).backward()
            if i[0] % SIGNAL_EVERY == 0:
                dpcs.collect_signals(loss, model)
            scaler.step(opt); scaler.update()
        else:
            loss.backward()
            if i[0] % SIGNAL_EVERY == 0:
                dpcs.collect_signals(loss, model)
            opt.step()
        i[0] += 1
        dpcs.end_step(opt, scaler)
        return loss.detach()

    avg_ms, ips, max_peak, last_loss = measure(step)
    modes = Counter(s.mode for s in dpcs._registry.values())
    return avg_ms, ips, max_peak, last_loss, dict(modes)

# --- Binary search: maximum tokens that fit (tokens = batch * seq) ---
def _safe_cuda_cleanup():
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()  # flush async errors first
        except Exception:
            pass
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()  # release cached blocks (won't increase allocatable memory)
        except Exception:
            # If a prior async error is bubbling here, just ignore and move on
            pass
    gc.collect()

def _is_oom(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return (
        isinstance(exc, getattr(torch.cuda, "OutOfMemoryError", RuntimeError))
        or "out of memory" in msg
        or "cuda error: out of memory" in msg
    )

def fits(tokens, use_dpcs_ckpt):
    """Try one fwd+bwd step with total tokens, return True if it fits (no OOM)."""
    batch = BATCH
    seq = max(tokens // batch, 1)

    _safe_cuda_cleanup()

    model = TinyTransformer().to(DEVICE)
    dpcs = None
    if use_dpcs_ckpt:
        dpcs = DPCS(device_type=DEVICE, signals_freq_steps=SIGNAL_EVERY)
        model = dpcs.wrap(model, allow_fp8=(DEVICE=="cuda"))
        dpcs._ckpt_policy = CheckpointPolicy.MUST_RECOMPUTE

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda") if DEVICE == "cuda" else None

    try:
        x = rand_batch(batch, seq)
        opt.zero_grad(set_to_none=True)

        if dpcs:
            dpcs.start_step()
            fwd_ctx, _ = dpcs.checkpoint_contexts()
            with fwd_ctx:
                with torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=(DEVICE=="cuda")):
                    y = model(x); loss = (y ** 2).mean()
        else:
            with torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=(DEVICE=="cuda")):
                y = model(x); loss = (y ** 2).mean()

        scaler.scale(loss).backward() if scaler else loss.backward()
        if dpcs: dpcs.collect_signals(loss, model)
        (scaler.step(opt), scaler.update()) if scaler else opt.step()
        if dpcs: dpcs.end_step(opt, scaler)
        return True

    except Exception as e:
        if _is_oom(e):
            return False
        raise

    finally:
        # Drop references and clean up aggressively between probes
        del model, opt
        if dpcs is not None: del dpcs
        for name in ("x", "y", "loss"):
            if name in locals(): del locals()[name]
        _safe_cuda_cleanup()

def max_tokens(lo=BATCH*128, hi=BATCH*8192, use_dpcs_ckpt=False):
    if DEVICE != "cuda":
        return None
    l, r = lo, hi
    while l < r:
        mid = (l + r + 1) // 2
        ok = fits(mid, use_dpcs_ckpt)
        print(f"[search {'DPCS' if use_dpcs_ckpt else 'BASE'}] try tokens={mid:,} -> {ok}")
        if ok: l = mid
        else:  r = mid - 1
    return l


'''
def fits(tokens, use_dpcs_ckpt):
    """Try one forward+backward step with total tokens, return True if it fits."""
    batch = BATCH
    seq = max(tokens // batch, 1)
    torch.cuda.empty_cache()
    model = TinyTransformer().to(DEVICE)
    dpcs = None
    if use_dpcs_ckpt:
        dpcs = DPCS(device_type=DEVICE, signals_freq_steps=SIGNAL_EVERY)
        model = dpcs.wrap(model, allow_fp8=(DEVICE=="cuda"))
        dpcs._ckpt_policy = CheckpointPolicy.MUST_RECOMPUTE

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda") if DEVICE == "cuda" else None

    try:
        x = rand_batch(batch, seq)
        opt.zero_grad(set_to_none=True)
        if dpcs: dpcs.start_step()
        if dpcs:
            fwd_ctx, _ = dpcs.checkpoint_contexts()
            with fwd_ctx:
                with torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=True):
                    y = model(x); loss = (y ** 2).mean()
        else:
            with torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=True):
                y = model(x); loss = (y ** 2).mean()
        scaler.scale(loss).backward() if scaler else loss.backward()
        if dpcs: dpcs.collect_signals(loss, model)
        (scaler.step(opt), scaler.update()) if scaler else opt.step()
        if dpcs: dpcs.end_step(opt, scaler)
        return True
    except RuntimeError as e:
        if "CUDA out of memory" in str(e): return False
        raise
        '''

def main():
    print(f"Device: {DEVICE}, CUDA available: {torch.cuda.is_available()}")
    if DEVICE == "cuda":
        free, total = torch.cuda.memory.mem_get_info()
        print(f"GPU Free/Total: {mb(free)}MB / {mb(total)}MB")

    # Throughput & peak memory at a fixed workload
    base_ms, base_ips, base_mem, base_loss = bench_baseline()
    noop_ms, noop_ips, noop_mem, noop_loss, noop_modes = bench_dpcs_noop()
    ck_ms, ck_ips, ck_mem, ck_loss, ck_modes = bench_dpcs_ckpt()

    print("\n=== Transformer throughput (avg steady-state) ===")
    print(f"Baseline AMP:    {base_ms:.2f} ms/step, {base_ips:.1f} tok/s, peak {mb(base_mem)} MB, last loss {base_loss:.4f}")
    print(f"DPCS (no-op):    {noop_ms:.2f} ms/step, {noop_ips:.1f} tok/s, peak {mb(noop_mem)} MB, last loss {noop_loss:.4f}, modes={noop_modes}")
    print(f"DPCS (ckpt ON):  {ck_ms:.2f} ms/step, {ck_ips:.1f} tok/s, peak {mb(ck_mem)} MB, last loss {ck_loss:.4f}, modes={ck_modes}")

    # Memory-bound demo: max tokens that fit, baseline vs DPCS with checkpointing
    if DEVICE == "cuda":
        base_max = max_tokens(use_dpcs_ckpt=False)
        dpcs_max = max_tokens(use_dpcs_ckpt=True)
        print("\n=== Max tokens that fit (batch * seq) ===")
        print(f"Baseline AMP max tokens: {base_max}")
        print(f"DPCS (ckpt ON)  max tokens: {dpcs_max}")

if __name__ == "__main__":
    main()
