"""Unit tests covering AMP overflow state machine behaviour.

The scenarios mirror the state diagram documented on :class:`PrecisionPolicy`:

  1. Clean steps that accumulate patience and demote to the low-precision mode.
  2. A single overflow that snaps back to ``fp32`` and engages cooldown.
  3. Repeated overflows that reset cooldown to prevent flip-flopping.
  4. NaN/Inf detection from ``GradScaler`` internals without an explicit scale drop.

The tests rely on a lightweight ``FakeScaler`` so no CUDA context is required.
"""

from __future__ import annotations

import os
import sys
from typing import List

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from dpcs.policies import PrecisionCfg, PrecisionPolicy  # noqa: E402
from dpcs.runtime import AmpOverflowMonitor  # noqa: E402
from dpcs.scheduler import DPCS  # noqa: E402


class FakeScaler:
    """Minimal ``GradScaler`` stand-in for overflow simulations."""

    def __init__(self, init_scale: float = 1024.0, enabled: bool = True) -> None:
        self._scale = float(init_scale)
        self._enabled = bool(enabled)
        self._next_scale = self._scale
        self._next_found = 0.0
        self._per_optimizer_states = {
            0: {"found_inf_per_device": {"cpu": 0.0}}
        }

    def is_enabled(self) -> bool:
        return self._enabled

    def get_scale(self) -> float:
        return float(self._scale)

    def schedule(self, *, drop: bool = False, found_inf: bool = False, scale: float | None = None) -> None:
        if scale is not None:
            self._next_scale = float(scale)
        elif drop:
            self._next_scale = max(self._scale / 2.0, 1.0)
        else:
            self._next_scale = self._scale
        self._next_found = 1.0 if found_inf else 0.0
        self._per_optimizer_states = {
            0: {"found_inf_per_device": {"cpu": 0.0}}
        }

    def update(self) -> None:
        self._scale = float(self._next_scale)
        self._per_optimizer_states = {
            0: {"found_inf_per_device": {"cpu": float(self._next_found)}}
        }
        # Keep subsequent updates idempotent unless ``schedule`` overrides them.
        self._next_scale = self._scale
        self._next_found = 0.0


def _reach_low_mode(policy: PrecisionPolicy, mode: str, steps: int, *, headroom: float = 0.5) -> str:
    for _ in range(steps):
        mode = policy.decide(headroom, None, None, False, mode)
    return mode


def test_amp_policy_clean_steps_follow_state_diagram() -> None:
    cfg = PrecisionCfg(prefer_bf16=False, patience=3, cooldown_steps=2)
    policy = PrecisionPolicy(cfg, bf16_supported=False, amp_available=True)
    mode = "fp32"
    states: List[str] = []

    for _ in range(cfg.patience):
        mode = policy.decide(0.6, None, None, False, mode)
        states.append(mode)

    assert states[:-1] == ["fp32"] * (cfg.patience - 1)
    assert states[-1] == policy.low_mode == "fp16"
    assert policy.cooldown == 0

    # Additional clean steps stay in the low mode per the state diagram.
    mode = policy.decide(0.6, None, None, False, mode)
    assert mode == policy.low_mode


def test_amp_policy_single_overflow_triggers_cooldown() -> None:
    cfg = PrecisionCfg(prefer_bf16=False, patience=3, cooldown_steps=2)
    policy = PrecisionPolicy(cfg, bf16_supported=False, amp_available=True)
    mode = _reach_low_mode(policy, "fp32", cfg.patience, headroom=0.7)
    assert mode == "fp16"

    scaler = FakeScaler(512.0)
    monitor = AmpOverflowMonitor(scaler, history=4)
    scaler.schedule(drop=True, found_inf=True)
    with monitor:
        scaler.update()

    assert monitor.last_overflow is True
    assert monitor.history[-1] is True

    mode = policy.decide(0.7, None, None, monitor.last_overflow, mode)
    assert mode == "fp32"
    assert policy.cooldown == cfg.cooldown_steps

    for _ in range(cfg.cooldown_steps):
        prev_cd = policy.cooldown
        mode = policy.decide(0.7, None, None, False, mode)
        assert mode == "fp32"
        assert policy.cooldown == max(prev_cd - 1, 0)

    rebound: List[str] = []
    for _ in range(cfg.patience):
        mode = policy.decide(0.7, None, None, False, mode)
        rebound.append(mode)

    assert rebound[:-1] == ["fp32"] * (cfg.patience - 1)
    assert rebound[-1] == policy.low_mode == "fp16"


def test_amp_policy_repeated_overflow_resets_cooldown() -> None:
    cfg = PrecisionCfg(prefer_bf16=False, patience=2, cooldown_steps=3)
    policy = PrecisionPolicy(cfg, bf16_supported=False, amp_available=True)
    mode = _reach_low_mode(policy, "fp32", cfg.patience, headroom=0.65)
    assert mode == "fp16"

    scaler = FakeScaler(256.0)
    monitor = AmpOverflowMonitor(scaler, history=5)

    # First overflow kicks us back to fp32.
    scaler.schedule(drop=True, found_inf=True)
    with monitor:
        scaler.update()
    mode = policy.decide(0.65, None, None, monitor.last_overflow, mode)
    assert mode == "fp32"
    assert policy.cooldown == cfg.cooldown_steps

    # Another overflow during cooldown should reset the timer and keep fp32.
    scaler.schedule(drop=True, found_inf=True)
    with monitor:
        scaler.update()
    mode = policy.decide(0.65, None, None, monitor.last_overflow, mode)
    assert mode == "fp32"
    assert policy.cooldown == cfg.cooldown_steps
    assert monitor.history.count(True) >= 2

    prev_cd = policy.cooldown
    mode = policy.decide(0.65, None, None, False, mode)
    assert mode == "fp32"
    assert policy.cooldown == max(prev_cd - 1, 0)

    # Drain cooldown and then accumulate patience to return to the low mode.
    while policy.cooldown > 0:
        mode = policy.decide(0.65, None, None, False, mode)
        assert mode == "fp32"

    back: List[str] = []
    for _ in range(cfg.patience):
        mode = policy.decide(0.65, None, None, False, mode)
        back.append(mode)

    assert back[:-1] == ["fp32"] * (cfg.patience - 1)
    assert back[-1] == policy.low_mode == "fp16"


def test_amp_policy_nan_inf_counts_as_overflow() -> None:
    cfg = PrecisionCfg(prefer_bf16=False, patience=2, cooldown_steps=2)
    policy = PrecisionPolicy(cfg, bf16_supported=False, amp_available=True)
    mode = _reach_low_mode(policy, "fp32", cfg.patience, headroom=0.8)
    assert mode == "fp16"

    scaler = FakeScaler(1024.0)
    monitor = AmpOverflowMonitor(scaler, history=4)
    scaler.schedule(found_inf=True)
    with monitor:
        scaler.update()

    assert monitor.last_overflow is True  # detected via found_inf without scale drop
    mode = policy.decide(0.8, None, None, monitor.last_overflow, mode)
    assert mode == "fp32"
    assert policy.cooldown == cfg.cooldown_steps

    # Clearing the flag should return to clean history.
    scaler.schedule(found_inf=False)
    with monitor:
        scaler.update()
    assert monitor.last_overflow is False


def test_scheduler_logs_overflow_fields() -> None:
    cfg = dict(
        device_type="cpu",
        enable_precision=1,
        precision_cfg=dict(prefer_bf16=False, patience=2, cooldown_steps=4),
    )
    scheduler = DPCS(**cfg)

    scaler = FakeScaler(256.0)
    monitor = scheduler.overflow_monitor(scaler)
    scaler.schedule(drop=True, found_inf=True)
    with monitor:
        scaler.update()

    class _StubLogger:
        def __init__(self) -> None:
            self.records: List[dict] = []

        def log(self, record):
            self.records.append(dict(record))

    stub = _StubLogger()
    scheduler._jsonl_logger = stub  # type: ignore[attr-defined]
    scheduler._log_every = 1

    scheduler.end_step(object(), scaler)

    assert stub.records, "Expected a JSONL record to be emitted"
    record = stub.records[-1]
    assert record["overflow_flag"] is True
    assert record["cooldown_remaining"] == scheduler._prec_pol.cooldown
    assert record["amp_mode"] == scheduler._amp_mode
