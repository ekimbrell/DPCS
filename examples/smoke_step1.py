import torch, torch.nn as nn
from dpcs import DPCS


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = nn.Sequential(nn.Linear(1024, 1024), nn.GELU(), nn.Linear(1024, 1024)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), 1e-3)
    scaler = torch.amp.GradScaler("cuda") if dev == "cuda" else None

    dpcs = DPCS(device_type=dev, signals_freq_steps=1, ckpt_low=0.10, ckpt_high=0.15)
    model = dpcs.wrap(model)

    for step in range(5):
        dpcs.start_step()
        opt.zero_grad(set_to_none=True)
        fwd_ctx, _ = dpcs.checkpoint_contexts_if_needed()
        x = torch.randn(8, 1024, device=dev)
        with fwd_ctx, torch.autocast(device_type=dev, dtype=torch.float16 if dev=="cuda" else torch.bfloat16, enabled=(dev!="cpu")):
            y = model(x); loss = (y**2).mean()
        if scaler:
            scaler.scale(loss).backward(); dpcs.collect_signals(loss, model)
            scaler.step(opt); scaler.update()
        else:
            loss.backward(); dpcs.collect_signals(loss, model); opt.step()
        dpcs.end_step(opt, scaler)
        print({"step": step+1, "ckpt_on": dpcs.is_checkpointing(), "modes": dpcs.modes_summary()})
    print("OK")

if __name__ == "__main__":
    main()
