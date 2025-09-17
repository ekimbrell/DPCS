import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from dpcs import DPCS

def make_model(d):
    return nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    ).to(d)

def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Keep checkpointing OFF for this smoke test; we'll validate it next step.
    dpcs = DPCS(
        device_type="cuda" if use_cuda else "cpu",
        enable_precision=True,           # CPU: False; CUDA: True
    )

    model = make_model(device)
    model = dpcs.wrap(model)

    opt = Adam(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler(enabled=dpcs.amp_uses_grad_scaler())

    B = 8
    for _ in range(2):
        dpcs.start_step()
        x = torch.randn(B, 128, device=device)
        y = torch.randint(0, 10, (B,), device=device)

        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        dpcs.collect_signals(loss, model)

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()

        dpcs.end_step(opt, scaler)

    mix = dpcs.modes_summary()
    assert isinstance(mix, dict)
    print("OK", mix)

if __name__ == "__main__":
    torch.manual_seed(0)
    main()
