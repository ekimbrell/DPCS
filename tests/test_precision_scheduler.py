# tests/test_precision_scheduler.py
import math
import os
import pytest
import torch
import torch.nn as nn

# Make sure src/ is importable when running "pytest" from repo root
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from dpcs import DPCS  # noqa: E402


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for AMP autocast/GradScaler path")
def test_precision_scheduler_cooldown_and_recovery():
    torch.manual_seed(0)
    dev = "cuda"

    # Tiny MLP (two Linear layers) so wrapping picks them up
    model = nn.Sequential(
        nn.Linear(1024, 1024),
        nn.GELU(),
        nn.Linear(1024, 1024),
    ).to(dev)

    # Enable scheduler; make it eager to choose low precision when stable:
    # - large epsilon_g => gvar_ema < eps_g almost always true (drops precision)
    # - modest kappa    => curvature proxy rarely trips fallback on its own
    cooldown = 3
    dpcs = DPCS(
        device_type=dev,
        enable_precision=True,
        epsilon_g=1e9,
        kappa=1e-1,
        cooldown_steps=cooldown,
        ema_beta=0.9,
    )
    model = dpcs.wrap(model)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler(device=dev)

    # Helper to run one scaled step and collect signals
    def one_step(inject_nan: bool = False):
        dpcs.start_step()
        opt.zero_grad(set_to_none=True)
        x = torch.randn(8, 1024, device=dev)
        with torch.autocast(device_type=dev, dtype=torch.float16, enabled=True):
            y = model(x)
            loss = (y ** 2).mean()
        scaler.scale(loss).backward()

        if inject_nan:
            # Inject a non-finite gradient into the *first wrapped module* to trigger fallback
            wrapped = [m for m in model.modules() if hasattr(m, "_dpcs_orig_forward")]
            assert wrapped, "No modules were wrapped by DPCS; expected at least one Linear."
            first = wrapped[0]
            # Make sure grad exists, then poison it
            p = next(p for p in first.parameters() if p.grad is not None)
            p.grad.view(-1)[0] = float("nan")

        dpcs.collect_signals(loss, model)
        scaler.step(opt)
        scaler.update()
        dpcs.end_step(opt, scaler)

    # ---- Step A: warm step (no NaNs) → should prefer FP16 where safe
    one_step(inject_nan=False)
    mix_a = dpcs.precision_mix()
    assert sum(mix_a.values()) > 0
    # We *expect* FP16 to be present because epsilon_g is huge (drop precision path)
    assert "fp16" in mix_a

    # ---- Step B: inject NaNs to force FP32 cooldown on targeted module
    one_step(inject_nan=True)
    mix_b = dpcs.precision_mix()
    # At least one module must be in fp32 after the overflow detection
    assert "fp32" in mix_b and mix_b["fp32"] >= 1

    # Grab that module's state to check the cooldown counter
    # (Find any module that’s currently fp32)
    target = None
    for mod, st in dpcs._registry.items():  # internal access OK for a white-box unit test
        if st.mode == "fp32":
            target = (mod, st)
            break
    assert target is not None, "Expected at least one module in fp32 after NaN injection."
    _, st = target
    assert st.cool == cooldown, f"Expected cooldown={cooldown}, got {st.cool}"

    # ---- Step C: run 'cooldown' normal steps; mode should stay fp32 until cool reaches 0
    for i in range(cooldown):
        one_step(inject_nan=False)
        assert st.cool == max(cooldown - (i + 1), 0)
        # during cooldown, mode is pinned to fp32
        assert st.mode == "fp32"

    # ---- Step D: after cooldown expires, with epsilon_g huge the module should drop to fp16 again
    one_step(inject_nan=False)
    assert st.cool == 0
    assert st.mode in ("fp16", "fp8", "fp32")
    # If FP8 is not enabled, fp16 is the expected low-precision target
    assert st.mode != "fp32", "Expected module to return to low precision after cooldown and stable grads."


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for AMP autocast/GradScaler path")
def test_precision_mix_sums_to_wrapped_count():
    torch.manual_seed(123)
    dev = "cuda"

    model = nn.Sequential(
        nn.Linear(512, 512),
        nn.GELU(),
        nn.Linear(512, 512),
    ).to(dev)

    dpcs = DPCS(
        device_type=dev,
        enable_precision=True,
        epsilon_g=1e9,
        kappa=1e-1,
        cooldown_steps=2,
    )
    model = dpcs.wrap(model)
    opt = torch.optim.SGD(model.parameters(), lr=1e-2)
    scaler = torch.amp.GradScaler(device=dev)

    # one step
    dpcs.start_step()
    opt.zero_grad(set_to_none=True)
    x = torch.randn(4, 512, device=dev)
    with torch.autocast(device_type=dev, dtype=torch.float16, enabled=True):
        y = model(x)
        loss = (y ** 2).mean()
    scaler.scale(loss).backward()
    dpcs.collect_signals(loss, model)
    scaler.step(opt); scaler.update()
    dpcs.end_step(opt, scaler)

    # mix accounting
    mix = dpcs.precision_mix()
    wrapped = [m for m in model.modules() if hasattr(m, "_dpcs_orig_forward")]
    assert sum(mix.values()) == len(wrapped)
    # Keys limited to supported modes
    for k in mix.keys():
        assert k in {"fp16", "fp32", "fp8"}
