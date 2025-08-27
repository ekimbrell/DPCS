import math
import time
import random
import pytest
import torch
import torch.nn as nn

from dpcs import DPCS


def set_seed(seed=0):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DeepMLP(nn.Module):
    """Activation-heavy MLP to magnify checkpointing effects."""
    def __init__(self, width=8192, depth=10, out_dim=1000):
        super().__init__()
        layers = []
        layers.append(nn.Linear(width, width))
        for _ in range(depth - 1):
            layers += [nn.GELU(), nn.Linear(width, width)]
        layers += [nn.GELU(), nn.Linear(width, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _clone_with_same_weights(model: nn.Module) -> nn.Module:
    clone = type(model)()
    clone.load_state_dict(model.state_dict())
    return clone


def _force_fp32_modes(sched: DPCS):
    for st in sched._registry.values():
        st.mode = "fp32"


@pytest.mark.parametrize("device", ["cpu"])  # CUDA memory test is separate
def test_checkpointing_equivalence_numeric_cpu(device):
    """
    Check that enabling checkpointing doesn't change loss/grad numerics (within tolerance)
    when precision scheduling is held at FP32.
    """
    set_seed(42)
    width, depth, batch, classes = 1024, 6, 8, 100

    model_base = DeepMLP(width=width, depth=depth, out_dim=classes).to(device)
    model_ckpt = DeepMLP(width=width, depth=depth, out_dim=classes).to(device)
    model_ckpt.load_state_dict(model_base.state_dict())

    x = torch.randn(batch, width, device=device)
    y = torch.randint(0, classes, (batch,), device=device)

    # schedulers: disable precision changes, force FP32 modes
    sched_base = DPCS(device_type=device, enable_precision=False)
    sched_ckpt = DPCS(device_type=device, enable_precision=False)

    model_base = sched_base.wrap(model_base)
    model_ckpt = sched_ckpt.wrap(model_ckpt)

    _force_fp32_modes(sched_base)
    _force_fp32_modes(sched_ckpt)

    # Force checkpointing only for ckpt run
    sched_base._ckpt_on = False
    sched_ckpt._ckpt_on = True

    crit = nn.CrossEntropyLoss()

    # BASELINE
    model_base.zero_grad(set_to_none=True)
    logits_base = model_base(x)
    loss_base = crit(logits_base, y)
    loss_base.backward()

    # CHECKPOINTED
    model_ckpt.zero_grad(set_to_none=True)
    logits_ckpt = model_ckpt(x)
    loss_ckpt = crit(logits_ckpt, y)
    loss_ckpt.backward()

    # Compare losses and a representative gradient
    assert torch.allclose(loss_base, loss_ckpt, rtol=1e-6, atol=1e-6)

    # Compare first linear weight grad
    lin_base = next(m for m in model_base.modules() if isinstance(m, nn.Linear))
    lin_ckpt = next(m for m in model_ckpt.modules() if isinstance(m, nn.Linear))
    assert torch.allclose(lin_base.weight.grad, lin_ckpt.weight.grad, rtol=1e-6, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for memory regression check")
def test_checkpointing_reduces_peak_memory_cuda():
    """
    Compare peak CUDA memory (allocator) between baseline and checkpointed runs.
    We hold precision at FP32 to isolate the effect of checkpointing.
    Success criterion: checkpointed peak memory must be at least 5% lower.
    """
    set_seed(123)
    device = "cuda"
    width, depth, batch, classes = 8192, 10, 64, 1000

    model_base = DeepMLP(width=width, depth=depth, out_dim=classes).to(device)
    model_ckpt = DeepMLP(width=width, depth=depth, out_dim=classes).to(device)
    model_ckpt.load_state_dict(model_base.state_dict())

    x = torch.randn(batch, width, device=device)
    y = torch.randint(0, classes, (batch,), device=device)

    sched_base = DPCS(device_type=device, enable_precision=False)
    sched_ckpt = DPCS(device_type=device, enable_precision=False)

    model_base = sched_base.wrap(model_base)
    model_ckpt = sched_ckpt.wrap(model_ckpt)

    _force_fp32_modes(sched_base)
    _force_fp32_modes(sched_ckpt)

    # Force checkpointing states explicitly
    sched_base._ckpt_on = False
    sched_ckpt._ckpt_on = True

    crit = nn.CrossEntropyLoss().to(device)

    # --- Baseline memory ---
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model_base.zero_grad(set_to_none=True)
    logits = model_base(x)
    loss = crit(logits, y)
    loss.backward()

    peak_baseline = torch.cuda.max_memory_allocated(device)

    # --- Checkpointing memory ---
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model_ckpt.zero_grad(set_to_none=True)
    logits = model_ckpt(x)
    loss = crit(logits, y)
    loss.backward()

    peak_ckpt = torch.cuda.max_memory_allocated(device)

    # Sanity: numbers are positive and comparable
    assert peak_baseline > 0 and peak_ckpt > 0

    # Expect at least 5% reduction with checkpointing
    reduction = (peak_baseline - peak_ckpt) / float(peak_baseline)
    assert reduction >= 0.05, f"Expected >=5% mem reduction, got {reduction*100:.2f}% (baseline={peak_baseline}, ckpt={peak_ckpt})"

    # Also ensure numerical parity within reasonable tolerance (GPU)
    # Recompute both losses fresh to compare
    model_base.zero_grad(set_to_none=True)
    model_ckpt.zero_grad(set_to_none=True)
    lb = crit(model_base(x), y)
    lc = crit(model_ckpt(x), y)
    assert torch.allclose(lb, lc, rtol=1e-5, atol=1e-6)
