"""Policies for precision selection and activation checkpointing.

This module isolates the *decision logic* from the runtime mechanics. It has no
PyTorch dependencies in the hot path (pure Python & math), so the scheduler can
call it without pulling extra overhead into the step loop.

Exported API:
  - PrecisionPolicy(cfg, bf16_supported: bool)
      .decide(headroom: float | None,
              grad_var: float | None,
              curvature: float | None,
              overflow: bool,
              current: str) -> str

  - CheckpointPolicy(cfg, ckpt_cfg)
      .plan(act_bytes_ema: list[float], headroom: float | None,
            free_bytes: int | None, peak_bytes: int | None) -> set[int]
      .blacklist(bad_ids: list[int], penalty_steps: int = 50) -> None
      .tick() -> None   # decay blacklist counters per step

The scheduler remains the orchestrator: it computes signals, passes them here,
and applies the returned decisions in batched form.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Set
import math

try:  # type: ignore
    from .config import DPCSConfig, CheckpointCfg
except Exception:  # pragma: no cover
    from config import DPCSConfig, CheckpointCfg  # type: ignore


# ----------------------------- small helpers ------------------------------

def _clamp01(x: float | None) -> float:
    if x is None:
        return 0.0
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


# ----------------------------- Precision policy ---------------------------

@dataclass(frozen=True)
class PrecisionCfg:
    """Lightweight knobs for the conservative precision policy."""

    prefer_bf16: bool = True
    patience: int = 8
    cooldown_steps: int = 2
    bf16_entry_headroom: float = 0.18
    bf16_exit_headroom: float = 0.08

    def __post_init__(self) -> None:
        object.__setattr__(self, "prefer_bf16", bool(self.prefer_bf16))
        object.__setattr__(self, "patience", max(1, int(self.patience)))
        object.__setattr__(self, "cooldown_steps", max(1, int(self.cooldown_steps)))
        entry = _clamp01(self.bf16_entry_headroom)
        exit_ = _clamp01(self.bf16_exit_headroom)
        if exit_ > entry:
            entry, exit_ = exit_, entry
        object.__setattr__(self, "bf16_entry_headroom", entry)
        object.__setattr__(self, "bf16_exit_headroom", exit_)


class PrecisionPolicy:
    """Finite-state policy for AMP mode selection.

    Modes: "fp32", "bf16", "fp16", and optionally "fp8" when TransformerEngine
    replacements are available. The conservative policy starts in fp32, waits
    for ``patience`` stable steps, and then demotes to the preferred low-precision
    mode (bf16 when available, otherwise fp16). Any overflow observation triggers
    an immediate promotion back to fp32 with a short cooldown window to reassess
    stability. FP8 decisions are gated by the runtime via :meth:`update_fp8_support`.

    State diagram::

        fp32 --(stable ≥ patience)--> low_mode
        low_mode --(overflow)--> fp32 [cooldown]
        fp32 --(cooldown > 0)--> fp32

    The cooldown ensures a run of fp32 steps before demotion resumes, preventing
    mode flip-flops when overflows occur in close succession.
    """

    def __init__(self, cfg: PrecisionCfg, bf16_supported: bool, amp_available: bool = True) -> None:
        self.cfg = cfg
        self.amp_available = bool(amp_available)
        self.bf16_supported = bool(bf16_supported)
        self.low_mode = "fp32"
        if self.amp_available:
            if self.bf16_supported and self.cfg.prefer_bf16:
                self.low_mode = "bf16"
            else:
                self.low_mode = "fp16"
        self.cooldown = 0
        patience = max(1, int(self.cfg.patience))
        self._cooldown_window = max(1, int(self.cfg.cooldown_steps))
        self._stable_steps = 0
        self._active_low_mode = self.low_mode
        # FP8 gating (configured lazily by the scheduler)
        self._fp8_supported = False
        self._fp8_low_headroom = 0.05
        self._fp8_high_headroom = 0.10
        self._fp8_reentry_cooldown = 0

    def update_fp8_support(
        self,
        supported: bool,
        low_headroom: Optional[float] = None,
        high_headroom: Optional[float] = None,
    ) -> None:
        """Toggle FP8 support and refresh headroom thresholds if provided."""

        self._fp8_supported = bool(supported)
        if low_headroom is not None:
            self._fp8_low_headroom = _clamp01(float(low_headroom))
        if high_headroom is not None:
            self._fp8_high_headroom = _clamp01(float(high_headroom))
        if self._fp8_high_headroom < self._fp8_low_headroom:
            self._fp8_low_headroom, self._fp8_high_headroom = (
                self._fp8_high_headroom,
                self._fp8_low_headroom,
            )
        if not self._fp8_supported:
            self._fp8_reentry_cooldown = 0

    def decide(
        self,
        headroom: Optional[float],
        grad_var: Optional[float],
        curvature: Optional[float],
        overflow: bool,
        current: str,
    ) -> str:
        del grad_var, curvature  # unused in the conservative policy

        if self._fp8_reentry_cooldown > 0:
            self._fp8_reentry_cooldown -= 1

        mode = str(current).lower()
        if mode not in {"fp32", "bf16", "fp16", "fp8"}:
            mode = "fp32"

        if overflow:
            self.cooldown = self._cooldown_window
            self._stable_steps = 0
            if self.low_mode == "bf16" and self.amp_available:
                self._active_low_mode = "fp16"
            if mode == "fp8":
                self._fp8_reentry_cooldown = max(self._fp8_reentry_cooldown, self._cooldown_window)
            return "fp32"

        if not self.amp_available or self.low_mode == "fp32":
            self.cooldown = 0
            self._stable_steps = 0
            return "fp32"

        fp8_supported = self._fp8_supported and self.amp_available
        hr = None if headroom is None else float(headroom)

        if self.low_mode == "bf16" and self.amp_available and hr is not None:
            if self._active_low_mode == "bf16" and hr <= self.cfg.bf16_exit_headroom:
                self._active_low_mode = "fp16"
            elif self._active_low_mode == "fp16" and hr >= self.cfg.bf16_entry_headroom:
                self._active_low_mode = "bf16"

        if not fp8_supported and mode == "fp8":
            mode = "fp32"

        if fp8_supported:
            if mode == "fp8":
                if hr is not None and hr >= self._fp8_high_headroom:
                    mode = "fp32"
                    self._fp8_reentry_cooldown = max(
                        self._fp8_reentry_cooldown, self._cooldown_window
                    )
                    self._stable_steps = 0
                else:
                    self.cooldown = 0
                    self._stable_steps = min(self._stable_steps + 1, self.cfg.patience)
                    return "fp8"
            elif hr is not None and hr <= self._fp8_low_headroom and self._fp8_reentry_cooldown == 0:
                self.cooldown = 0
                self._stable_steps = 0
                self._fp8_reentry_cooldown = max(
                    self._fp8_reentry_cooldown, self._cooldown_window
                )
                return "fp8"

        if mode != "fp32" and mode not in {"bf16", "fp16"}:
            mode = "fp32"

        if mode == "fp32":
            if self.cooldown > 0:
                self.cooldown -= 1
                self._stable_steps = 0
                return "fp32"
            self._stable_steps = min(self._stable_steps + 1, self.cfg.patience)
            if self._stable_steps >= self.cfg.patience:
                if self.low_mode == "bf16":
                    return self._active_low_mode
                return self.low_mode
            return "fp32"

        # Already running in low precision. Stay there unless a new overflow occurs.
        self.cooldown = 0
        self._stable_steps = min(self._stable_steps + 1, self.cfg.patience)
        if self.low_mode == "bf16":
            return self._active_low_mode
        return self.low_mode


# --------------------------- Checkpointing policy --------------------------

class CheckpointPolicy:
    """Selects a subset of leaves to checkpoint under memory pressure.

    The policy exposes three primitives:
      - ``plan``: rank leaves by activation-byte EMA and choose a top-K under
        current memory pressure.
      - ``blacklist``: temporarily prevent harmful ids from being re-enabled.
      - ``tick``: decay blacklist counters each step.
    """

    def __init__(self, cfg: DPCSConfig, ckpt_cfg: CheckpointCfg) -> None:
        self.cfg = cfg
        self.ckpt_cfg = ckpt_cfg
        self._blacklist: dict[int, int] = {}  # id -> remaining penalty steps
        self._pat_cnt: int = 0
        self._last_target: Set[int] = set()

    # ---- blacklist management ----
    def blacklist(self, bad_ids: Iterable[int], penalty_steps: int = 50) -> None:
        pen = max(1, int(penalty_steps))
        for i in bad_ids:
            self._blacklist[int(i)] = pen

    def tick(self) -> None:
        if not self._blacklist:
            return
        dead = []
        for k in list(self._blacklist.keys()):
            self._blacklist[k] -= 1
            if self._blacklist[k] <= 0:
                dead.append(k)
        for k in dead:
            self._blacklist.pop(k, None)

    # ---- core decision ----
    def _compute_pressure(
        self, headroom: Optional[float], free_bytes: Optional[int], peak_bytes: Optional[int]
    ) -> float:
        pressure = 0.0
        if headroom is not None:
            hr = _clamp01(headroom)
            lo = float(self.cfg.low_headroom_frac)
            hi = float(self.cfg.hi_headroom_frac)
            if hr <= lo:
                pressure = 1.0
            elif hr < hi:
                span = max(hi - lo, 1e-6)
                pressure = max(pressure, (hi - hr) / span)
        if free_bytes is not None and peak_bytes is not None and peak_bytes > 0:
            ratio = float(free_bytes) / max(float(peak_bytes), 1.0)
            ratio = max(0.0, min(ratio, 1.0))
            pressure = max(pressure, 1.0 - ratio)
        return max(0.0, min(pressure, 1.0))

    def plan(
        self,
        act_bytes: List[float],
        headroom: Optional[float],
        free_bytes: Optional[int],
        peak_bytes: Optional[int],
    ) -> Set[int]:
        n = len(act_bytes)
        if n == 0:
            self._last_target = set()
            return self._last_target

        pressure = self._compute_pressure(headroom, free_bytes, peak_bytes)
        cap = max(0.0, min(float(self.ckpt_cfg.max_fraction), 1.0))
        cfg_cap = float(self.cfg.ckpt_topk_frac)
        if cfg_cap > 0.0:
            cap = min(cap, cfg_cap)
        target_frac = cap * pressure
        k = int(math.ceil(target_frac * n))
        if k <= 0:
            self._last_target = set()
            return self._last_target

        scores = [float(b) for b in act_bytes]

        min_bytes = int(self.cfg.min_activation_bytes_to_ckpt)
        candidates = [
            i
            for i, b in enumerate(act_bytes)
            if float(b) >= min_bytes and (i not in self._blacklist)
        ]
        if not candidates:
            self._last_target = set()
            return self._last_target

        candidates.sort(key=lambda i: scores[i], reverse=True)
        chosen = set(candidates[:k])

        if self._pat_cnt < max(1, int(self.cfg.ckpt_patience)):
            self._pat_cnt += 1
            return self._last_target

        self._pat_cnt = 0
        self._last_target = chosen
        return chosen


__all__ = ["PrecisionCfg", "PrecisionPolicy", "CheckpointPolicy"]
