# examples/bench_dpcs_adaptive.py
print("[bench] loaded file")  # early sanity print

import time
from collections import Counter
import torch
import torch.nn as nn
from dpcs import DPCS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DIM, DEPTH, BATCH = 3072, 8, 64
DIM, DEPTH, BATCH = 4096, 12, 128
STEPS, WARMUP = 90, 30
CKPT_HEADROOM_THRESH = 0.90
SIGNAL_EVERY = 200

def _is_oom(e: Exception) -> bool:
    s = str(e).lower()
    try:
        import torch
        return isinstance(e, torch.cuda.OutOfMemoryError) or ("out of memory" in s)
    except Exception:
        return ("out of memory" in s)

class MLP(nn.Module):
    def __init__(self, dim, depth):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers += [nn.Linear(dim, dim), nn.GELU()]
        layers += [nn.Linear(dim, dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

def make_model(seed=0):
    torch.manual_seed(seed)
    return MLP(DIM, DEPTH).to(DEVICE)

def headroom_ratio():
    if DEVICE == "cuda":
        free, total = torch.cuda.memory.mem_get_info()
        return free / total
    return 1.0

def measure(run_fn, steps=STEPS, warmup=WARMUP):
    times_ms, max_peak, last_loss = [], 0, 0.0
    for i in range(steps):
        if DEVICE == "cuda":
            torch.cuda.reset_peak_memory_stats()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            loss = run_fn()
            end.record()
            torch.cuda.synchronize()
            if i >= warmup:
                times_ms.append(start.elapsed_time(end))  # ms
                max_peak = max(max_peak, torch.cuda.max_memory_allocated())
        else:
            t0 = time.perf_counter(); loss = run_fn()
            if i >= warmup: times_ms.append((time.perf_counter() - t0) * 1000.0)
        last_loss = float(loss)
    avg_ms = sum(times_ms) / max(len(times_ms), 1)
    ips = BATCH / (avg_ms / 1000.0) if avg_ms > 0 else 0.0
    return avg_ms, ips, max_peak, last_loss

def mb(x): return int(x / (1024**2)) if x else 0

def bench_baseline():
    model = make_model(seed=123)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda") if DEVICE == "cuda" else None
    x = torch.randn(BATCH, DIM, device=DEVICE)
    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=DEVICE, dtype=(torch.float16 if DEVICE=="cuda" else torch.bfloat16), enabled=True):
            y = model(x); loss = y.pow(2).mean()
        if scaler: scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        else: loss.backward(); opt.step()
        return loss.detach()
    return measure(step)

def bench_dpcs_noop():
    model = make_model(seed=123)
    dpcs = DPCS(epsilon_g=1e-3, kappa=5.0, device_type=DEVICE, signals_freq_steps=10_000)  # signals basically off
    model = dpcs.wrap(model, allow_fp8=(DEVICE == "cuda"))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda") if DEVICE == "cuda" else None
    x = torch.randn(BATCH, DIM, device=DEVICE)
    def step():
        dpcs.start_step()
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=DEVICE, dtype=(torch.float16 if DEVICE=="cuda" else torch.bfloat16), enabled=True):
            y = model(x); loss = y.pow(2).mean()
        if scaler: scaler.scale(loss).backward(); dpcs.collect_signals(loss, model); scaler.step(opt); scaler.update()
        else: loss.backward(); dpcs.collect_signals(loss, model); opt.step()
        dpcs.end_step(opt, scaler); return loss.detach()
    avg_ms, ips, max_peak, last_loss = measure(step)
    modes = Counter(s.mode for s in dpcs._registry.values())
    return avg_ms, ips, max_peak, last_loss, dict(modes)

def bench_dpcs_adaptive():
    model = make_model(seed=123)
    dpcs = DPCS(epsilon_g=1e-3, kappa=5.0, device_type=DEVICE, signals_freq_steps=SIGNAL_EVERY)
    model = dpcs.wrap(model, allow_fp8=(DEVICE == "cuda"))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda") if DEVICE == "cuda" else None
    x = torch.randn(BATCH, DIM, device=DEVICE)
    ckpt_on_steps = 0
    def step(i=[0]):
        dpcs.start_step()
        opt.zero_grad(set_to_none=True)
        use_ckpt = (DEVICE == "cuda" and headroom_ratio() < 0.12)
        if use_ckpt:
            fwd_ctx, _ = dpcs.checkpoint_contexts()
            with fwd_ctx:
                with torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=True):
                    y = model(x); loss = y.pow(2).mean()
            nonlocal ckpt_on_steps; ckpt_on_steps += 1
        else:
            with torch.autocast(device_type=DEVICE, dtype=(torch.float16 if DEVICE=="cuda" else torch.bfloat16), enabled=True):
                y = model(x); loss = y.pow(2).mean()
        do_signals = (i[0] % SIGNAL_EVERY == 0); i[0] += 1
        if scaler: scaler.scale(loss).backward(); 
        else: loss.backward()
        if do_signals: dpcs.collect_signals(loss, model)
        if scaler: scaler.step(opt); scaler.update()
        else: opt.step()
        dpcs.end_step(opt, scaler); return loss.detach()
    avg_ms, ips, max_peak, last_loss = measure(step)
    modes = Counter(s.mode for s in dpcs._registry.values())
    return avg_ms, ips, max_peak, last_loss, dict(modes), ckpt_on_steps

def max_batch_search(lo=8, hi=8192, use_dpcs=False):
    if DEVICE != "cuda": return None
    def fits(batch):
        torch.cuda.empty_cache()
        model = make_model(seed=456)
        dpcs = None
        if use_dpcs:
            dpcs = DPCS(device_type=DEVICE, signals_freq_steps=SIGNAL_EVERY)
            model = dpcs.wrap(model, allow_fp8=True)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler("cuda")
        try:
            x = torch.randn(batch, DIM, device=DEVICE)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=True):
                y = model(x); loss = y.pow(2).mean()
            scaler.scale(loss).backward()
            if dpcs: dpcs.collect_signals(loss, model)
            scaler.step(opt); scaler.update()
            if dpcs: dpcs.end_step(opt, scaler)
            return True
        except Exception as e:
            if _is_oom(e):
                return False
            raise
        
    l, r = lo, hi
    while l < r:
        mid = (l + r + 1) // 2
        if fits(mid): l = mid
        else: r = mid - 1
    return l

def main():
    print(f"[bench] Device: {DEVICE}, CUDA available: {torch.cuda.is_available()}")
    base_ms, base_ips, base_mem, base_loss = bench_baseline()
    noop_ms, noop_ips, noop_mem, noop_loss, noop_modes = bench_dpcs_noop()
    ada_ms, ada_ips, ada_mem, ada_loss, ada_modes, ckpt_steps = bench_dpcs_adaptive()

    def mb(x): return int(x / (1024**2)) if x else 0
    print("\n=== Throughput (avg over steady-state steps) ===")
    print(f"Baseline AMP:      {base_ms:.2f} ms/step, {base_ips:.1f} img/s, peak {mb(base_mem)} MB, last loss {base_loss:.4f}")
    print(f"DPCS (no-op):      {noop_ms:.2f} ms/step, {noop_ips:.1f} img/s, peak {mb(noop_mem)} MB, last loss {noop_loss:.4f}, modes={noop_modes}")
    print(f"DPCS (adaptive):   {ada_ms:.2f} ms/step, {ada_ips:.1f} img/s, peak {mb(ada_mem)} MB, last loss {ada_loss:.4f}, modes={ada_modes}, ckpt_steps={ckpt_steps}")

    if DEVICE == "cuda":
        base_max = max_batch_search(use_dpcs=False)
        dpcs_max = max_batch_search(use_dpcs=True)
        print("\n=== Max batch that fits (binary search) ===")
        print(f"Baseline AMP max batch: {base_max}")
        print(f"DPCS adaptive max batch: {dpcs_max}")

if __name__ == "__main__":
    main()
