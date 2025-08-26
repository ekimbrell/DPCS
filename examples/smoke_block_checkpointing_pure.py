import torch, torch.nn as nn
from torch.utils.checkpoint import checkpoint_sequential

def make_block(in_out=2048, hidden=4096):
    # A moderately heavy MLP block
    return nn.Sequential(
        nn.Linear(in_out, hidden), nn.GELU(),
        nn.Linear(hidden, hidden), nn.GELU(),
        nn.Linear(hidden, in_out)
    )

class DeepMLP(nn.Module):
    def __init__(self, depth=6, in_out=2048, hidden=4096):
        super().__init__()
        self.blocks = nn.Sequential(*[make_block(in_out, hidden) for _ in range(depth)])

    def forward_plain(self, x):
        return self.blocks(x)

    def forward_ckpt(self, x, segments=2):
        # IMPORTANT: pass x as the required 'input' arg (3rd positional)
        # and explicit use_reentrant=False per current guidance.
        return checkpoint_sequential(self.blocks, segments, x, use_reentrant=False)

def measure_step(model: nn.Module, model_fwd, dev="cuda", autocast_dtype=torch.float16, B=8, D=2048):
    x = torch.randn(B, D, device=dev)
    torch.cuda.reset_peak_memory_stats()

    # ---- forward-only peak ----
    with torch.autocast(device_type=dev, dtype=autocast_dtype, enabled=(dev=="cuda")):
        y = model_fwd(x); loss = (y**2).mean()
    torch.cuda.synchronize()
    fwd_peak_mb = int(torch.cuda.max_memory_allocated()/1024/1024)

    # ---- full step peak (after backward) ----
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    scaler = torch.amp.GradScaler("cuda") if dev=="cuda" else None

    torch.cuda.reset_peak_memory_stats()
    if scaler:
        with torch.autocast(device_type=dev, dtype=autocast_dtype, enabled=True):
            y = model_fwd(x); loss = (y**2).mean()
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    else:
        with torch.autocast(device_type=dev, dtype=torch.bfloat16, enabled=False):
            y = model_fwd(x); loss = (y**2).mean()
        loss.backward(); opt.step()
    torch.cuda.synchronize()
    full_peak_mb = int(torch.cuda.max_memory_allocated()/1024/1024)

    return fwd_peak_mb, full_peak_mb

if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA required for this smoke"
    torch.backends.cudnn.benchmark = False
    dev = "cuda"
    model = DeepMLP(depth=6).to(dev)

    # No checkpoint
    fwd_no, full_no = measure_step(model, model.forward_plain, dev)

    # With checkpoint (2 segments = checkpoint blocks, not leaves)
    fwd_ck, full_ck = measure_step(model, lambda x: model.forward_ckpt(x, segments=2), dev)

    print(f"forward-only peak (no ckpt): {fwd_no} MB")
    print(f"forward-only peak (with ckpt): {fwd_ck} MB")
    print(f"full-step peak   (no ckpt):   {full_no} MB")
    print(f"full-step peak   (with ckpt): {full_ck} MB")
