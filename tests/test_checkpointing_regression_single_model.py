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


def _force_precision_mode(sched: DPCS, mode: str = "fp32"):
    sched.force_precision(mode)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for memory regression check")
@pytest.mark.parametrize("baseline_mode", ["bf16", "fp16"])
def test_checkpointing_reduces_peak_memory_cuda_single_model(baseline_mode):
    """
    Single-model memory regression:
    - Toggle checkpointing OFF/ON on the same wrapped model to avoid contamination
      from a second model's gradients.
    - Use BF16/FP16 baselines to mirror precision-disabled training.
    Success criterion: checkpointing peak memory at least 5% lower than baseline.
    """
    if baseline_mode == "bf16":
        try:
            if not torch.cuda.is_bf16_supported():
                pytest.skip("CUDA device does not support bfloat16")
        except Exception:
            pytest.skip("Unable to query bfloat16 support")
    set_seed(123)
    device = "cuda"
    width, depth, batch, classes = 8192, 10, 64, 1000

    model = DeepMLP(width=width, depth=depth, out_dim=classes).to(device)
    sched = DPCS(device_type=device, enable_precision=False)
    model = sched.wrap(model)
    _force_precision_mode(sched, baseline_mode)

    x = torch.randn(batch, width, device=device)
    y = torch.randint(0, classes, (batch,), device=device)
    crit = nn.CrossEntropyLoss().to(device)

    # --- BASELINE (checkpointing OFF) ---
    sched._ckpt_on = False
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()

    model.zero_grad(set_to_none=True)
    logits_base = model(x)
    loss_base = crit(logits_base, y)
    loss_base.backward()
    torch.cuda.synchronize()

    peak_baseline = torch.cuda.max_memory_allocated(device)

    # Clear grads to truly free their memory before the next phase
    for p in model.parameters():
        p.grad = None
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()

    # --- CHECKPOINTED (checkpointing ON) ---
    sched._ckpt_on = True

    model.zero_grad(set_to_none=True)
    logits_ckpt = model(x)
    loss_ckpt = crit(logits_ckpt, y)
    loss_ckpt.backward()
    torch.cuda.synchronize()

    peak_ckpt = torch.cuda.max_memory_allocated(device)

    # Sanity: numbers are positive
    assert peak_baseline > 0 and peak_ckpt > 0

    # Expect at least 5% reduction with checkpointing
    reduction = (peak_baseline - peak_ckpt) / float(peak_baseline)
    assert reduction >= 0.05, (
        f"Expected >=5% mem reduction, got {reduction*100:.2f}% "
        f"(baseline={peak_baseline}, ckpt={peak_ckpt})"
    )

    # Numerical parity of loss
    assert torch.allclose(loss_base, loss_ckpt, rtol=5e-3, atol=1e-5)
