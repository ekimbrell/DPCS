import random
import pytest
import torch
import torch.nn as nn

from dpcs import DPCS

# --- Version-portable SDPA "math" forcing -----------------------------------
try:
    # New API: pass allowed backends as a set/list
    from torch.nn.attention import sdpa_kernel, SDPBackend

    def force_math_sdpa():
        return sdpa_kernel([SDPBackend.MATH])
except Exception:  # pragma: no cover - fallback path for older Torch
    # Deprecated API: boolean flags on CUDA backend
    from torch.backends.cuda import sdp_kernel as _old_sdpa_kernel

    def force_math_sdpa():
        return _old_sdpa_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)


def set_seed(seed=0):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TinyTransformer(nn.Module):
    """
    Transformer encoder stack tuned for activation-dominant behavior.
    Dropout is 0.0 for deterministic numeric equivalence.
    """
    def __init__(self, d_model=512, nhead=8, dim_ff=2048, nlayers=6, n_classes=1000, dropout=0.0):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=nlayers)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x):  # x: [B, S, D]
        h = self.enc(x)
        h = h.mean(dim=1)  # simple pool instead of CLS
        return self.head(h)


def _force_fp32_modes(sched: DPCS):
    for st in sched._registry.values():
        st.mode = "fp32"


@pytest.mark.parametrize("device", ["cpu"])  # CPU numeric equivalence
def test_checkpointing_equivalence_numeric_cpu_transformer(device):
    set_seed(123)
    B, S, D, C = 4, 64, 128, 50
    model_a = TinyTransformer(d_model=D, nhead=4, dim_ff=256, nlayers=2, n_classes=C, dropout=0.0).to(device)
    model_b = TinyTransformer(d_model=D, nhead=4, dim_ff=256, nlayers=2, n_classes=C, dropout=0.0).to(device)
    model_b.load_state_dict(model_a.state_dict())

    x = torch.randn(B, S, D, device=device)
    y = torch.randint(0, C, (B,), device=device)

    sched_a = DPCS(device_type=device, enable_precision=False, wrap_types=(nn.Linear, nn.TransformerEncoderLayer))
    sched_b = DPCS(device_type=device, enable_precision=False, wrap_types=(nn.Linear, nn.TransformerEncoderLayer))
    model_a = sched_a.wrap(model_a)
    model_b = sched_b.wrap(model_b)
    _force_fp32_modes(sched_a)
    _force_fp32_modes(sched_b)

    # baseline (no ckpt)
    sched_a._ckpt_on = False
    crit = nn.CrossEntropyLoss()
    model_a.zero_grad(set_to_none=True)
    la = crit(model_a(x), y)
    la.backward()

    # checkpointed
    sched_b._ckpt_on = True
    model_b.zero_grad(set_to_none=True)
    lb = crit(model_b(x), y)
    lb.backward()

    assert torch.allclose(la, lb, rtol=1e-6, atol=1e-6)
    # Compare a representative grad (final head weight)
    assert torch.allclose(
        next(model_a.head.parameters()).grad,
        next(model_b.head.parameters()).grad,
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for memory regression check")
def test_checkpointing_reduces_peak_memory_cuda_transformer_single_model():
    """
    Single-model CUDA memory regression on a Transformer encoder stack.
    Wrap entire TransformerEncoderLayer so attention is checkpointed too.
    Use a version-portable SDPA context to force 'math' backend for clearer deltas.
    Success: checkpointing reduces peak allocator bytes by >=3%.
    """
    set_seed(321)
    device = "cuda"
    # Activation-heavy but safe defaults; adjust if OOM or too tiny
    B, S, D, C = 8, 1024, 512, 1000
    model = TinyTransformer(d_model=D, nhead=8, dim_ff=2048, nlayers=6, n_classes=C, dropout=0.0).to(device)

    sched = DPCS(device_type=device, enable_precision=False, wrap_types=(nn.Linear, nn.TransformerEncoderLayer))
    model = sched.wrap(model)
    _force_fp32_modes(sched)

    x = torch.randn(B, S, D, device=device)
    y = torch.randint(0, C, (B,), device=device)
    crit = nn.CrossEntropyLoss().to(device)

    def run_and_measure(ckpt_on: bool):
        sched._ckpt_on = ckpt_on
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        model.zero_grad(set_to_none=True)
        with force_math_sdpa():
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

    # Verify numerical parity
    assert torch.allclose(loss_base, loss_ckpt, rtol=1e-5, atol=1e-6)
