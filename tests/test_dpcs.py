import math
import types
import pytest
import torch
import torch.nn as nn

# Import the scheduler from the local file
from dpcs import DPCS, DPCSConfig


class TinyMLP(nn.Module):
    def __init__(self, d_in=32, d_hidden=64, d_out=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x):
        return self.net(x)


def _one_step(model, sched: DPCS, device: str = "cpu"):
    model.train()
    x = torch.randn(8, 32, device=device)
    y = torch.randint(0, 10, (8,), device=device)
    optim = torch.optim.SGD(model.parameters(), lr=1e-2)
    scaler = torch.amp.GradScaler(device) if (device == "cuda" and torch.cuda.is_available()) else None

    sched.start_step()
    with torch.autocast(device, dtype=torch.float16 if device == "cuda" else torch.bfloat16, enabled=True):
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
    if scaler is not None:
        scaler.scale(loss).backward()
    else:
        loss.backward()

    sched.collect_signals(loss, model)

    if scaler is not None:
        scaler.step(optim)
        scaler.update()
    else:
        optim.step()
    optim.zero_grad(set_to_none=True)

    sched.end_step(optim, scaler)
    return loss.detach()


def test_wrap_and_modes_basic_cpu():
    device = "cpu"
    model = TinyMLP().to(device)
    sched = DPCS(device_type=device)
    model = sched.wrap(model)

    # Ensure modules were registered
    assert len(sched.modes_summary()) >= 1

    # Run a step to exercise policy
    _one_step(model, sched, device=device)

    mix = sched.modes_summary()
    # At least one of these modes should be present
    assert any(k in mix for k in ("fp16", "fp32", "fp8"))


def test_cooldown_on_overflow_cpu():
    device = "cpu"
    model = TinyMLP().to(device)
    sched = DPCS(device_type=device, cooldown_steps=2)
    model = sched.wrap(model)

    # Do a normal step first
    _one_step(model, sched, device=device)

    # Force non-finite grad on the first wrapped module to trigger cooldown
    wrapped_mod = next(iter(sched._registry.keys()))
    for p in wrapped_mod.parameters(recurse=False):
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        p.grad.data.fill_(float("inf"))

    # Collect and apply decisions
    sched.collect_signals(torch.tensor(0.0), model)
    sched.end_step(torch.optim.SGD(model.parameters(), 1e-2), None)

    st = sched._registry[wrapped_mod]
    assert st.mode == "fp32"
    assert st.cool >= 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for FP8 test")
@pytest.mark.skipif(
    not getattr(__import__('dpcs', fromlist=['DPCS']), 'te', None),
    reason="Transformer Engine not available",
)
def test_fp8_mode_path_smoke_cuda():
    device = "cuda"
    model = TinyMLP().to(device)
    sched = DPCS(device_type=device, allow_fp8=True, swap_modules_for_fp8=True,
                 epsilon_g=1.0,  # very high threshold so variance < eps is likely
                 kappa=1e9)      # avoid curvature trigger
    model = sched.wrap(model)

    # Manually set low variance on all modules to force drop→fp8
    for st in sched._registry.values():
        st.gvar_ema = 0.0

    _one_step(model, sched, device=device)
    mix = sched.modes_summary()
    assert mix.get("fp8", 0) >= 1


def test_checkpointing_hysteresis_toggle():
    # Use ckpt_need=1 so a single vote toggles
    device = "cpu"
    model = TinyMLP().to(device)
    sched = DPCS(device_type=device, ckpt_low=0.04, ckpt_high=0.3, ckpt_need=1)
    model = sched.wrap(model)

    # Force headroom low → ON
    sched._headroom = 0.01
    sched.end_step(torch.optim.SGD(model.parameters(), 1e-2), None)
    assert sched.is_checkpointing() is True

    # Force headroom high → OFF
    sched._headroom = 0.9
    sched.end_step(torch.optim.SGD(model.parameters(), 1e-2), None)
    assert sched.is_checkpointing() is False
