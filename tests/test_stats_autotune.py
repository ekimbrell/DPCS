import math
import random
import pytest
import torch
import torch.nn as nn

from dpcs import DPCS


# ------- helpers -------------------------------------------------------------

def set_seed(seed=0):
    random.seed(seed)
    torch.manual_seed(seed)


def _concat_param_grads(mod: nn.Module) -> torch.Tensor:
    flats = []
    for p in mod.parameters(recurse=False):
        if p.grad is None:
            continue
        flats.append(p.grad.detach().reshape(-1).to(dtype=torch.float64))
    if not flats:
        return torch.zeros(0, dtype=torch.float64)
    return torch.cat(flats)


class TwoLinear(nn.Module):
    def __init__(self, d_in=7, d_h=5, d_out=3):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_h)
        self.fc2 = nn.Linear(d_h, d_out)

    def forward(self, x):
        return self.fc2(torch.tanh(self.fc1(x)))


# ------- tests ---------------------------------------------------------------

@pytest.mark.parametrize("device", ["cpu"])  # pure CPU, deterministic
def test_welford_variance_matches_reference(device):
    """Per-module Welford variance equals reference population variance of all param grads."""
    set_seed(123)
    model = TwoLinear().to(device)
    cfg = dict(device_type=device, enable_precision=False, wrap_types=(nn.Linear,))
    dpcs = DPCS(**cfg)
    model = dpcs.wrap(model)

    # Single deterministic backward
    x = torch.randn(16, 7, device=device)
    y = torch.randn(16, 3, device=device)
    loss = nn.MSELoss()(model(x), y)
    loss.backward()

    # Collect signals (computes Welford variance into st.last_var_step)
    dpcs.collect_signals(loss, model)

    # For each wrapped module, compare last_var_step vs reference var on concatenated grads
    for m, st in dpcs._registry.items():
        ref = _concat_param_grads(m)
        assert ref.numel() > 0
        var_ref = (ref.var(unbiased=False)).item()
        assert st.last_var_step is not None
        # Tight tolerance: both use float32 internally; allow small epsilon
        assert math.isfinite(st.last_var_step)
        assert abs(st.last_var_step - var_ref) <= max(1e-10, 1e-6 * (1.0 + abs(var_ref)))


@pytest.mark.parametrize("device", ["cpu"])  # pure CPU, deterministic
def test_gvar_ema_updates_correctly(device):
    """EMA(gvar) follows beta * old + (1-beta) * new with correct initialization."""
    set_seed(7)
    model = TwoLinear().to(device)
    beta = 0.8
    cfg = dict(device_type=device, enable_precision=False, wrap_types=(nn.Linear,), ema_beta=beta)
    dpcs = DPCS(**cfg)
    model = dpcs.wrap(model)

    # Step 1
    dpcs.start_step()
    x1 = torch.randn(8, 7, device=device)
    y1 = torch.randn(8, 3, device=device)
    loss1 = nn.MSELoss()(model(x1), y1)
    loss1.backward()
    dpcs.collect_signals(loss1, model)

    # Capture per-module new var (step1)
    step1_vars = {}
    for m, st in dpcs._registry.items():
        assert st.last_var_step is not None
        step1_vars[m] = st.last_var_step
        # First EMA assignment should seed to new value
        assert st.gvar_ema is not None
        assert abs(st.gvar_ema - st.last_var_step) <= 1e-8 + 1e-6 * (1 + abs(st.last_var_step))

    # Zero grads before next step
    model.zero_grad(set_to_none=True)

    # Step 2 (different grads)
    dpcs.start_step()
    x2 = torch.randn(8, 7, device=device)
    y2 = torch.randn(8, 3, device=device)
    loss2 = nn.MSELoss()(model(x2), y2)
    loss2.backward()
    dpcs.collect_signals(loss2, model)

    # EMA check: gvar_ema == beta * step1 + (1-beta) * step2
    for m, st in dpcs._registry.items():
        v1 = step1_vars[m]
        v2 = st.last_var_step
        assert v2 is not None
        expected = beta * v1 + (1 - beta) * v2
        assert st.gvar_ema is not None
        assert abs(st.gvar_ema - expected) <= 1e-8 + 1e-5 * (1 + abs(expected))


@pytest.mark.parametrize("device", ["cpu"])  # pure CPU, deterministic
def test_autotune_thresholds_after_warmup(device):
    """
    Warm-up auto-tuner: after N steps, epsilon_g and kappa should equal the configured
    percentiles of the pooled per-step module samples (variance & curvature),
    clamped by minimums.
    """
    set_seed(99)
    warm_steps = 3
    gq = 0.5  # median
    cq = 0.5  # median
    cfg = dict(
        device_type=device,
        enable_precision=False,
        wrap_types=(nn.Linear,),
        ema_beta=0.7,
        autotune_precision=True,
        autotune_warmup_steps=warm_steps,
        autotune_gvar_percentile=gq,
        autotune_curv_percentile=cq,
        autotune_min_eps=1e-12,
        autotune_min_kappa=1e-12,
    )
    dpcs = DPCS(**cfg)
    model = dpcs.wrap(TwoLinear().to(device))

    # Pools to compute our own percentiles
    gvars = []
    curvs = []

    crit = nn.MSELoss()

    # Run warm-up steps
    for step in range(1, warm_steps + 1):
        dpcs.start_step()
        x = torch.randn(10, 7, device=device)
        y = torch.randn(10, 3, device=device)
        loss = crit(model(x), y)
        loss.backward()
        dpcs.collect_signals(loss, model)

        # Accumulate per-module samples exactly as DPCS does
        for m, st in dpcs._registry.items():
            assert st.last_var_step is not None
            assert st.last_curv_step is not None
            gvars.append(float(st.last_var_step))
            curvs.append(float(st.last_curv_step))

        # Prepare for next iteration
        model.zero_grad(set_to_none=True)

        # Trigger autotune on boundary
        dpcs.end_step(optim=torch.optim.SGD(model.parameters(), lr=0.0), scaler=None)

    # After warm-up boundary, thresholds should be updated
    # Compute our own percentiles using torch.quantile to mirror implementation
    t_g = torch.tensor(gvars, dtype=torch.float64)
    t_c = torch.tensor(curvs, dtype=torch.float64)
    eps_ref = torch.quantile(t_g, torch.tensor([gq], dtype=torch.float64)).item()
    kap_ref = torch.quantile(t_c, torch.tensor([cq], dtype=torch.float64)).item()

    assert math.isfinite(dpcs.cfg.epsilon_g) and math.isfinite(dpcs.cfg.kappa)
    # Compare with small relative tolerance
    assert abs(dpcs.cfg.epsilon_g - eps_ref) <= 1e-12 + 1e-6 * (1 + abs(eps_ref))
    assert abs(dpcs.cfg.kappa - kap_ref) <= 1e-12 + 1e-6 * (1 + abs(kap_ref))

    # Sanity: thresholds should be non-trivial (greater than minima)
    assert dpcs.cfg.epsilon_g >= 1e-12
    assert dpcs.cfg.kappa >= 1e-12
