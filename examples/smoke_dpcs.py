# examples/smoke_dpcs.py
import torch, torch.nn as nn
from dpcs import DPCS

device = "cuda" if torch.cuda.is_available() else "cpu"
m = nn.Sequential(nn.Linear(1024,1024), nn.GELU(), nn.Linear(1024,1024)).to(device)

dpcs = DPCS(epsilon_g=1e-3, kappa=5.0, device_type=device, fp8_backend="te")
m = dpcs.wrap(m, allow_fp8=True)

opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
scaler = torch.amp.GradScaler("cuda") if device=="cuda" else None

for step in range(10):
    dpcs.start_step(); opt.zero_grad(set_to_none=True)
    with dpcs.checkpoint_contexts_if_needed()[0], \
         torch.autocast(device_type=device, dtype=(torch.float16 if device=="cuda" else torch.bfloat16), enabled=True):
        x = torch.randn(32,1024, device=device); y = m(x); loss = (y**2).mean()
    (scaler.scale(loss).backward(), scaler.step(opt), scaler.update()) if scaler else (loss.backward(), opt.step())
    dpcs.collect_signals(loss, m)
    dpcs.end_step(opt, scaler)
    print(dpcs.summary())
