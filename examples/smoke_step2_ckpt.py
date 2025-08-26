import torch, torch.nn as nn
from dpcs import DPCS

def fwd_peak_mb():
    return int(torch.cuda.max_memory_allocated() / 1024 / 1024)

def measure_once(use_ckpt: bool):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = False
    m = nn.Sequential(
        nn.Linear(2048, 4096), nn.GELU(),
        nn.Linear(4096, 4096), nn.GELU(),
        nn.Linear(4096, 2048)
    ).to(dev)
    opt = torch.optim.SGD(m.parameters(), 1e-3)
    scaler = torch.amp.GradScaler("cuda") if dev=="cuda" else None
    dpcs = DPCS(device_type=dev, signals_freq_steps=1, ckpt_low=0.0, ckpt_high=1.0)  # we'll drive the gate manually
    m = dpcs.wrap(m)

    # Manually force gate for this demo
    dpcs._ckpt_on = use_ckpt

    # One training step with forward-only peak instrumentation
    dpcs.start_step()
    opt.zero_grad(set_to_none=True)
    x = torch.randn(8, 2048, device=dev)

    fwd_ctx, _ = dpcs.checkpoint_contexts_if_needed()

    torch.cuda.reset_peak_memory_stats()
    with fwd_ctx, torch.autocast(device_type=dev, dtype=torch.float16 if dev=="cuda" else torch.bfloat16, enabled=(dev!="cpu")):
        y = m(x); loss = (y**2).mean()
    torch.cuda.synchronize()
    fwd_peak = fwd_peak_mb()

    if scaler:
        scaler.scale(loss).backward(); dpcs.collect_signals(loss, m)
        scaler.step(opt); scaler.update()
    else:
        loss.backward(); dpcs.collect_signals(loss, m); opt.step()
    dpcs.end_step(opt, scaler)
    return fwd_peak

if __name__ == "__main__":
    assert torch.cuda.is_available()
    a = measure_once(use_ckpt=False)
    b = measure_once(use_ckpt=True)
    print(f"forward-only peak (no ckpt): {a} MB")
    print(f"forward-only peak (with ckpt): {b} MB")
