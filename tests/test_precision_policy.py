import os
import sys

import pytest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from dpcs.policies import PrecisionCfg, PrecisionPolicy  # noqa: E402


@pytest.mark.parametrize("bf16_supported", [True, False])
def test_policy_demotes_after_patience(bf16_supported: bool) -> None:
    cfg = PrecisionCfg(patience=3)
    pol = PrecisionPolicy(cfg, bf16_supported=bf16_supported, amp_available=True)
    mode = "fp32"

    for _ in range(cfg.patience - 1):
        mode = pol.decide(None, None, None, False, mode)
        assert mode == "fp32"

    mode = pol.decide(None, None, None, False, mode)
    assert mode == pol.low_mode
    mode = pol.decide(None, None, None, False, mode)
    assert mode == pol.low_mode


def test_policy_overflow_forces_cooldown() -> None:
    cfg = PrecisionCfg(patience=6)
    pol = PrecisionPolicy(cfg, bf16_supported=True, amp_available=True)
    mode = "fp32"

    for _ in range(cfg.patience):
        mode = pol.decide(None, None, None, False, mode)
    assert mode == "bf16"

    mode = pol.decide(None, None, None, True, mode)
    assert mode == "fp32"
    assert pol.cooldown == 2

    while pol.cooldown > 0:
        prev_cd = pol.cooldown
        mode = pol.decide(None, None, None, False, mode)
        assert mode == "fp32"
        assert pol.cooldown == max(prev_cd - 1, 0)

    for _ in range(cfg.patience - 1):
        mode = pol.decide(None, None, None, False, mode)
        assert mode == "fp32"

    mode = pol.decide(None, None, None, False, mode)
    assert mode == "bf16"


def test_policy_stays_fp32_when_amp_unavailable() -> None:
    cfg = PrecisionCfg(patience=2)
    pol = PrecisionPolicy(cfg, bf16_supported=True, amp_available=False)
    mode = "fp32"

    for _ in range(5):
        mode = pol.decide(None, None, None, False, mode)
        assert mode == "fp32"


def test_policy_respects_fp16_preference() -> None:
    cfg = PrecisionCfg(prefer_bf16=False, patience=3)
    pol = PrecisionPolicy(cfg, bf16_supported=True, amp_available=True)
    mode = "fp32"

    for _ in range(cfg.patience):
        mode = pol.decide(None, None, None, False, mode)

    assert mode == "fp16"
