"""Unit tests for PrecisionPolicy overflow recovery logic."""

from __future__ import annotations

import os
import sys

import pytest



# Ensure ``src/`` layout is importable when running pytest from repo root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from dpcs.config import DPCSConfig  # noqa: E402
from dpcs.policies import PrecisionPolicy  # noqa: E402


def _low_mode(policy: PrecisionPolicy) -> str:
    return "bf16" if policy.bf16_supported else "fp16"


def _drain_cooldown(policy: PrecisionPolicy, current: str, *, headroom: float, grad_var: float, curvature: float) -> str:
    """Advance the policy through its cooldown while asserting it stays in fp32."""

    while policy.cooldown > 0:
        prev = policy.cooldown
        current = policy.decide(
            headroom=headroom,
            grad_var=grad_var,
            curvature=curvature,
            overflow=False,
            current=current,
        )
        assert current == "fp32"
        assert policy.cooldown == prev - 1
    return current


@pytest.mark.parametrize("bf16_supported", [True, False])
def test_precision_policy_leaves_fp32_after_cooldown_when_safe(bf16_supported: bool):
    cfg = DPCSConfig.from_kwargs(mode_patience=2)
    policy = PrecisionPolicy(cfg, bf16_supported=bf16_supported)
    current = _low_mode(policy)

    # --- Scenario 1: low headroom should force an immediate drop after cooldown ---
    current = policy.decide(
        headroom=0.5,
        grad_var=cfg.epsilon_g_high * 10.0,
        curvature=cfg.kappa_high * 10.0,
        overflow=True,
        current=current,
    )
    assert current == "fp32"

    # Trigger another overflow while still in cooldown to ensure repeated events extend it.
    current = policy.decide(
        headroom=0.5,
        grad_var=cfg.epsilon_g_high * 5.0,
        curvature=cfg.kappa_high * 5.0,
        overflow=True,
        current=current,
    )
    assert current == "fp32"

    current = _drain_cooldown(policy, current, headroom=0.5, grad_var=cfg.epsilon_g_high, curvature=cfg.kappa_high)

    # With the cooldown finished and low headroom, the policy should return to the lower mode immediately.
    current = policy.decide(
        headroom=cfg.low_headroom_frac * 0.5,
        grad_var=cfg.epsilon_g_low * 0.5,
        curvature=cfg.kappa_low * 0.5,
        overflow=False,
        current=current,
    )
    assert current == _low_mode(policy)
    assert policy._force_fp32_until_safe is False

    # --- Scenario 2: even with reasonable headroom, safe gradients/curvature should drop precision ---
    current = policy.decide(
        headroom=0.4,
        grad_var=cfg.epsilon_g_high * 5.0,
        curvature=cfg.kappa_high * 5.0,
        overflow=True,
        current=current,
    )
    assert current == "fp32"

    current = _drain_cooldown(policy, current, headroom=0.4, grad_var=cfg.epsilon_g_high, curvature=cfg.kappa_high)

    current = policy.decide(
        headroom=0.4,
        grad_var=cfg.epsilon_g_low * 0.25,
        curvature=cfg.kappa_low * 0.25,
        overflow=False,
        current=current,
    )
    assert current == _low_mode(policy)
    assert policy._force_fp32_until_safe is False


def test_precision_policy_stays_fp32_when_signals_stay_risky():
    cfg = DPCSConfig.from_kwargs(mode_patience=2)
    policy = PrecisionPolicy(cfg, bf16_supported=True)
    current = _low_mode(policy)

    # Trigger an overflow to enter the forced fp32 cooldown window.
    current = policy.decide(
        headroom=0.95,
        grad_var=cfg.epsilon_g_low,
        curvature=cfg.kappa_low,
        overflow=True,
        current=current,
    )
    assert current == "fp32"
    assert policy._force_fp32_until_safe is True

    # Drain the cooldown while keeping the signals in a "risky" regime
    # (ample headroom with high gradients/curvature).
    current = _drain_cooldown(
        policy,
        current,
        headroom=max(0.0, min(1.0, cfg.hi_headroom_frac + 0.1)),
        grad_var=cfg.epsilon_g_high * 2.0,
        curvature=cfg.kappa_high * 2.0,
    )
    assert policy.cooldown == 0
    assert policy._force_fp32_until_safe is True

    # Immediately after the cooldown the policy should remain in fp32 and clear
    # the overflow guard because the signals still strongly favor high
    # precision.
    current = policy.decide(
        headroom=cfg.hi_headroom_frac + 0.1,
        grad_var=cfg.epsilon_g_high * 2.0,
        curvature=cfg.kappa_high * 2.0,
        overflow=False,
        current=current,
    )
    assert current == "fp32"
    assert policy._force_fp32_until_safe is False

    # Subsequent risky steps should continue to prefer fp32 without flipping
    # back to lower precision.
    current = policy.decide(
        headroom=cfg.hi_headroom_frac + 0.05,
        grad_var=cfg.epsilon_g_high * 1.5,
        curvature=cfg.kappa_high * 1.5,
        overflow=False,
        current=current,
    )
    assert current == "fp32"
    assert policy._force_fp32_until_safe is False
