import math
import random
import pytest
import torch
import torch.nn as nn
from contextlib import contextmanager

from dpcs import DPCS

# --- SDPA helpers (version portable) -----------------------------------------
try:
    from torch.nn.attention import sdpa_kernel, SDPBackend  # PyTorch ≥ 2.1
    _SDPA_OK = True
except Exception:  # pragma: no cover
    _SDPA_OK = False

    @contextmanager
    def sdpa_kernel(*_args, **_kwargs):  # type: ignore[misc]
        yield

    class SDPBackend:  # type: ignore[override]
        MATH = "MATH"
        EFFICIENT_ATTENTION = "EFFICIENT_ATTENTION"
        FLASH_ATTENTION = "FLASH_ATTENTION"


def set_seed(seed=0):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --- tiny model: Transformer encoder + classifier ----------------------------
class TinyTransformer(nn.Module):
    def __init__(self, d_model=128, nhead=4, dim_ff=256, nlayers=2, n_classes=50, dropout=0.0):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
                                               dropout=dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.cls = nn.Linear(d_model, n_classes)

    def forward(self, x):
        h = self.enc(x)
        return self.cls(h[:, 0])


def _force_precision_mode(sched: DPCS, mode: str = "fp32"):
    # helper to keep numerics comparable
    sched.force_precision(mode)


@pytest.mark.parametrize("device", ["cpu"])  # numerical parity on CPU
def test_checkpointing_equivalence_numeric_cpu_transformer(device):
    set_seed(123)
    B, S, D, C = 4, 64, 128, 50
    model_a = TinyTransformer(d_model=D, nhead=4, dim_ff=256, nlayers=2, n_classes=C).to(device)
    model_b = TinyTransformer(d_model=D, nhead=4, dim_ff=256, nlayers=2, n_classes=C).to(device)
    model_b.load_state_dict(model_a.state_dict())

    x = torch.randn(B, S, D, device=device)
    y = torch.randint(0, C, (B,), device=device)

    # Use explicit SDPA backend in the scheduler; and in this test use the single-enum form
    print("[test] SDPA backend: MATH")

    sched_a = DPCS(device_type=device, enable_precision=False, sdpa_backends=(SDPBackend.MATH,))
    sched_b = DPCS(device_type=device, enable_precision=False, sdpa_backends=(SDPBackend.MATH,))
    model_a = sched_a.wrap(model_a)
    model_b = sched_b.wrap(model_b)
    _force_precision_mode(sched_a)
    _force_precision_mode(sched_b)

    # baseline
    sched_a._ckpt_on = False
    crit = nn.CrossEntropyLoss()
    model_a.zero_grad(set_to_none=True)
    with sdpa_kernel(SDPBackend.MATH):
        la = crit(model_a(x), y)
    la.backward()

    # checkpointed
    sched_b._ckpt_on = True
    model_b.zero_grad(set_to_none=True)
    with sdpa_kernel(SDPBackend.MATH):
        lb = crit(model_b(x), y)
    lb.backward()

    assert torch.allclose(la, lb, rtol=1e-6, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for memory regression check")
@pytest.mark.parametrize("baseline_mode", ["bf16", "fp16"])
def test_checkpointing_reduces_peak_memory_cuda_transformer_single_model(baseline_mode):
    """
    Single-model CUDA memory regression on a Transformer encoder stack.
    Wrap entire TransformerEncoderLayer so attention is checkpointed too.
    Parameterized across BF16/FP16 baselines to mirror precision-disabled runs.
    Force 'math' SDPA backend for clearer deltas.
    Success: checkpointing reduces peak allocator bytes by >=3%.
    """
    set_seed(321)
    device = "cuda"
    # Activation-heavy but safe defaults; adjust if OOM or too tiny
    B, S, D, C = 8, 1024, 512, 1000
    model = TinyTransformer(d_model=D, nhead=8, dim_ff=2048, nlayers=6, n_classes=C, dropout=0.0).to(device)

    print("[test] SDPA backend: MATH")

    sched = DPCS(device_type=device, enable_precision=False,
                 wrap_types=(nn.Linear, nn.TransformerEncoderLayer),
                 sdpa_backends=(SDPBackend.MATH,), force_sdpa_in_blocks=True)
    model = sched.wrap(model)
    if baseline_mode == "bf16":
        try:
            if not torch.cuda.is_bf16_supported():
                pytest.skip("CUDA device does not support bfloat16")
        except Exception:
            pytest.skip("Unable to query bfloat16 support")
    _force_precision_mode(sched, baseline_mode)

    x = torch.randn(B, S, D, device=device)
    y = torch.randint(0, C, (B,), device=device)
    crit = nn.CrossEntropyLoss().to(device)

    def run_and_measure(ckpt_on: bool):
        sched._ckpt_on = ckpt_on
        sched.start_step()  # ensure top-K selection built
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        model.zero_grad(set_to_none=True)
        with sdpa_kernel(SDPBackend.MATH):
            loss = crit(model(x), y)
            loss.backward()
        torch.cuda.synchronize()
        return loss, torch.cuda.max_memory_allocated(device)

    # BASELINE (no checkpoint)
    loss_base, peak_base = run_and_measure(False)

    # drop grads fully before ckpt pass
    for p in model.parameters():
        p.grad = None
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()

    # CHECKPOINTED
    loss_ckpt, peak_ckpt = run_and_measure(True)

    assert peak_base > 0 and peak_ckpt > 0
    reduction = (peak_base - peak_ckpt) / float(peak_base)
    assert reduction >= 0.03, (
        f"Expected >=3% mem reduction, got {reduction*100:.2f}% "
        f"(baseline={peak_base}, ckpt={peak_ckpt})"
    )

    # Also sanity check losses are close (GPU allows a bit looser tol)
    assert torch.allclose(loss_base, loss_ckpt, rtol=5e-3, atol=1e-5)
