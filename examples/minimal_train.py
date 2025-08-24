import torch, torch.nn as nn
from dpcs import DPCS

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = nn.Sequential(nn.Linear(1024, 1024), nn.GELU(), nn.Linear(1024, 1024)).to(device)

    dpcs = DPCS(epsilon_g=1e-3, kappa=5.0, fp8_backend="te")
    model = dpcs.wrap(model)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler(device if device != "cpu" else None)

    for step in range(5):
        dpcs.start_step()
        opt.zero_grad(set_to_none=True)
        x = torch.randn(8, 1024, device=device)

        fwd_ctx, _ = dpcs.checkpoint_contexts()
        with fwd_ctx:
            with torch.autocast(device_type=device, enabled=device!="cpu"):
                y = model(x); loss = y.pow(2).mean()

        scaler.scale(loss).backward()
        dpcs.collect_signals(loss, model)
        scaler.step(opt); scaler.update()
        dpcs.end_step(opt, scaler)

if __name__ == "__main__":
    main()
