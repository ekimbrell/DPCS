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

    def __post_init__(self) -> None:
        object.__setattr__(self, "prefer_bf16", bool(self.prefer_bf16))
        object.__setattr__(self, "patience", max(1, int(self.patience)))


class PrecisionPolicy:
    """Finite-state policy for AMP mode selection.

    Modes: "fp32", "bf16", "fp16". FP8 support is intentionally not wired for
    the conservative "v0" policy; once the system has gathered enough stable
    steps it demotes from fp32 to the preferred low-precision mode (bf16 when
    available, otherwise fp16). Any overflow observation triggers an immediate
    promotion back to fp32 with a short cooldown window to reassess stability.
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
        half = patience // 2
        self._cooldown_window = max(1, min(2, max(1, half)))
        self._stable_steps = 0

    def decide(
        self,
        headroom: Optional[float],
        grad_var: Optional[float],
        curvature: Optional[float],
        overflow: bool,
        current: str,
    ) -> str:
        del headroom, grad_var, curvature  # unused in the conservative policy
        if not self.amp_available or self.low_mode == "fp32":
            self.cooldown = 0
            self._stable_steps = 0
            return "fp32"

        mode = str(current).lower()
        if overflow:
            self.cooldown = self._cooldown_window
            self._stable_steps = 0
            return "fp32"

        if mode != "fp32" and mode not in {"bf16", "fp16"}:
            mode = "fp32"

        if mode == "fp32":
            if self.cooldown > 0:
                self.cooldown -= 1
                self._stable_steps = 0
                return "fp32"
            self._stable_steps = min(self._stable_steps + 1, self.cfg.patience)
            if self._stable_steps >= self.cfg.patience:
                return self.low_mode
            return "fp32"

        # Already running in low precision. Stay there unless a new overflow occurs.
        self.cooldown = 0
        self._stable_steps = min(self._stable_steps + 1, self.cfg.patience)
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


__all__ = ["PrecisionPolicy", "CheckpointPolicy"]
