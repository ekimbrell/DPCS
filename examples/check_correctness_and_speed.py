# examples/check_correctness_and_speed.py
import torch, torch.nn as nn, time
from dpcs import DPCS

DEV = "cuda" if torch.cuda.is_available() else "cpu"
B,S,D = 32, 1024, 1024
model = nn.Sequential(nn.Linear(D,D), nn.GELU(), nn.Linear(D,D)).to(DEV)

def run(step_fn, steps=50, warmup=10):
    times=[]; peak=0; last=0.0
    for i in range(steps):
        if DEV=="cuda":
            torch.cuda.reset_peak_memory_stats()
            st, ed = torch.cuda.Event(True), torch.cuda.Event(True)
            st.record(); loss = step_fn(); ed.record()
            torch.cuda.synchronize()
            if i>=warmup: times.append(st.elapsed_time(ed)); peak=max(peak, torch.cuda.max_memory_allocated())
        else:
            t0=time.perf_counter(); loss=step_fn()
            if i>=warmup: times.append((time.perf_counter()-t0)*1000)
        last=float(loss)
    return sum(times)/len(times), peak, last

# Baseline AMP
opt = torch.optim.AdamW(model.parameters(), 1e-3)
sc  = torch.amp.GradScaler("cuda") if DEV=="cuda" else None
x   = torch.randn(B,D, device=DEV)

def step_amp():
    opt.zero_grad(set_to_none=True)
    with torch.autocast(device_type=DEV, dtype=(torch.float16 if DEV=="cuda" else torch.bfloat16), enabled=True):
        y = model(x); loss=(y**2).mean()
    (sc.scale(loss).backward(), sc.step(opt), sc.update()) if sc else (loss.backward(), opt.step())
    return loss.detach()

base_ms, base_peak, base_loss = run(step_amp)

# DPCS adaptive (precision+ckpt)
model2 = nn.Sequential(nn.Linear(D,D), nn.GELU(), nn.Linear(D,D)).to(DEV)
dpcs   = DPCS(device_type=DEV, epsilon_g=1e-3, kappa=5.0, fp8_backend="te")
model2 = dpcs.wrap(model2, allow_fp8=True)
opt2   = torch.optim.AdamW(model2.parameters(), 1e-3)
sc2    = torch.amp.GradScaler("cuda") if DEV=="cuda" else None
x2     = torch.randn(B,D, device=DEV)

def step_dpcs():
    dpcs.start_step(); opt2.zero_grad(set_to_none=True)
    with dpcs.checkpoint_contexts_if_needed()[0], torch.autocast(device_type=DEV, dtype=(torch.float16 if DEV=="cuda" else torch.bfloat16), enabled=True):
        y = model2(x2); loss=(y**2).mean()
    if sc2: sc2.scale(loss).backward()
    else:   loss.backward()
    dpcs.collect_signals(loss, model2)
    (sc2.step(opt2), sc2.update()) if sc2 else opt2.step()
    dpcs.end_step(opt2, sc2)
    return loss.detach()

dpcs_ms, dpcs_peak, dpcs_loss = run(step_dpcs)

print(f"Baseline AMP: {base_ms:.1f} ms, peak {base_peak/1024/1024:.0f} MB")
print(f"DPCS adapt  : {dpcs_ms:.1f} ms, peak {dpcs_peak/1024/1024:.0f} MB")
