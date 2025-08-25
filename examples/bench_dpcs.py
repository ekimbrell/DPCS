import time
from collections import Counter
import torch
import torch.nn as nn

try:
    from dpcs import DPCS
except Exception as e:
    raise SystemExit(f"Import dpcs failed: {e}")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 64
DIM = 2048
STEPS = 80
WARMUP = 20

def make_model(seed=0):
    torch.manual_seed(seed)
    m = nn.Sequential(
        nn.Linear(DIM, DIM), nn.GELU(), nn.Linear(DIM, DIM)
    ).to(DEVICE)
    return m

def measure(run_fn, steps=STEPS, warmup=WARMUP):
    times_ms = []
    max_peak = 0
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
            t0 = time.perf_counter()
            loss = run_fn()
            if i >= warmup:
                times_ms.append((time.perf_counter() - t0) * 1000.0)
    avg_ms = sum(times_ms) / max(len(times_ms), 1)
    ips = BATCH / (avg_ms / 1000.0)
    return avg_ms, ips, max_peak, float(loss)

def run_baseline():
    model = make_model(seed=123)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda") if DEVICE == "cuda" else None
    x = torch.randn(BATCH, DIM, device=DEVICE)

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=DEVICE,
                            dtype=(torch.float16 if DEVICE == "cuda" else torch.bfloat16),
                            enabled=True):
            y = model(x); loss = y.pow(2).mean()
        if scaler:
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        else:
            loss.backward(); opt.step()
        return loss.detach()
    return measure(step)

def run_dpcs():
    model = make_model(seed=123)
    dpcs = DPCS(epsilon_g=1e-3, kappa=5.0, signals_freq_steps=50, device_type="cuda")
    # dpcs = DPCS(epsilon_g=1e-3, kappa=5.0, fp8_backend="te", device_type=DEVICE)
    model = dpcs.wrap(model, allow_fp8=(DEVICE == "cuda"))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda") if DEVICE == "cuda" else None
    x = torch.randn(BATCH, DIM, device=DEVICE)

    def step():
        dpcs.start_step()
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=DEVICE,
                            dtype=(torch.float16 if DEVICE == "cuda" else torch.bfloat16),
                            enabled=True):
            y = model(x); loss = y.pow(2).mean()
        if scaler:
            scaler.scale(loss).backward(); dpcs.collect_signals(loss, model)
            scaler.step(opt); scaler.update()
        else:
            loss.backward(); dpcs.collect_signals(loss, model)
            opt.step()
        dpcs.end_step(opt, scaler)
        return loss.detach()

    avg_ms, ips, max_peak, last_loss = measure(step)

    # Inspect how many modules were in each precision mode:
    try:
        modes = Counter(s.mode for s in dpcs._registry.values())
    except Exception:
        modes = {}
    return avg_ms, ips, max_peak, last_loss, modes

def main():
    print(f"Device: {DEVICE}, CUDA available: {torch.cuda.is_available()}")
    base_ms, base_ips, base_mem, base_loss = run_baseline()
    dpcs_ms, dpcs_ips, dpcs_mem, dpcs_loss, modes = run_dpcs()

    def mb(x): return int(x / (1024**2)) if x else 0
    print("\n=== Results (avg over steady-state steps) ===")
    print(f"Baseline AMP:   {base_ms:.2f} ms/step, {base_ips:.1f} img/s, peak {mb(base_mem)} MB, last loss {base_loss:.4f}")
    print(f"DPCS enabled:   {dpcs_ms:.2f} ms/step, {dpcs_ips:.1f} img/s, peak {mb(dpcs_mem)} MB, last loss {dpcs_loss:.4f}")
    if modes:
        print(f"DPCS precision mix (modules): {dict(modes)}")

if __name__ == "__main__":
    main()
