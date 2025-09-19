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


@dataclass
class _Patience:
    enter: int
    exit: int
    cnt: int = 0
    state: bool = False

    def update(self, want_on: bool) -> bool:
        """Hysteresis with separate enter/exit patience.
        Returns current state after update.
        """
        if want_on == self.state:
            self.cnt = 0
            return self.state
        self.cnt += 1
        need = self.enter if want_on else self.exit
        if self.cnt >= max(1, need):
            self.state = want_on
            self.cnt = 0
        return self.state


# ----------------------------- Precision policy ---------------------------

class PrecisionPolicy:
    """Finite-state policy for AMP mode selection.

    Modes: "fp32", "bf16", "fp16", optionally "fp8" when supported by the
    runtime helper. The policy still governs the global precision mode
    while the runtime swaps FP8-capable modules when requested.

    Signals considered:
      - ``headroom``: free/total VRAM fraction (0..1). Low headroom → lower precision.
      - ``grad_var``: EMA of gradient variance. Low → lower precision safe; high → promote.
      - ``curvature``: optional curvature proxy (e.g., power-iter HVP eigval). High → promote.
      - ``overflow``: GradScaler overflow → immediate fp32 cooldown.
    """

    def __init__(self, cfg: DPCSConfig, bf16_supported: bool, fp8_supported: bool = False) -> None:
        self.cfg = cfg
        self.bf16_supported = bool(bf16_supported)
        self.fp8_supported = bool(fp8_supported)
        self.cooldown = 0
        self.pat = _Patience(enter=max(1, cfg.mode_patience), exit=max(1, cfg.mode_patience))
        # When an overflow triggers a forced fp32 interval we keep track of it so
        # we can immediately return to a lower mode once conditions are safe
        # again (e.g., low headroom or tame gradients).
        self._force_fp32_until_safe = False

    def _lower_mode(self) -> str:
        # prefer fp8 when available, then bf16, else fp16
        if self.fp8_supported:
            return "fp8"
        if self.bf16_supported:
            return "bf16"
        return "fp16"

    def _promote_mode(self, current: str) -> str:
        if current == "fp32":
            return "fp32"
        if current == "fp8":
            if self.bf16_supported:
                return "bf16"
            return "fp32"
        if current == "bf16":
            return "fp32"
        if current == "fp16":
            return "fp32"
        return "fp32"

    def update_fp8_support(self, supported: bool) -> None:
        self.fp8_supported = bool(supported)

    def decide(
        self,
        headroom: Optional[float],
        grad_var: Optional[float],
        curvature: Optional[float],
        overflow: bool,
        current: str,
    ) -> str:
        # 1) Overflow → force fp32 for a cooldown window
        if overflow:
            self.cooldown = max(self.cooldown, 8)
            self._force_fp32_until_safe = True
            return "fp32"

        if self.cooldown > 0:
            self.cooldown -= 1
            return "fp32"

        # 2) Combine signals
        hr = _clamp01(headroom)
        gv = 0.0 if (grad_var is None) else float(grad_var)
        curv = 0.0 if (curvature is None) else float(curvature)
        low_headroom = hr < self.cfg.low_headroom_frac
        safe_signals = (
            (grad_var is None or gv <= self.cfg.epsilon_g_low)
            and (curvature is None or curv <= self.cfg.kappa_low)
        )


        want_low = low_headroom or (gv < self.cfg.epsilon_g_low and curv < self.cfg.kappa_low)
        want_high = (hr > self.cfg.hi_headroom_frac) or (gv > self.cfg.epsilon_g_high or curv > self.cfg.kappa_high)

        if self._force_fp32_until_safe:
            if low_headroom or safe_signals:
                # Drop back to the preferred lower mode immediately after the
                # overflow-induced cooldown finishes. Also reset patience so the
                # lower mode sticks unless high-pressure signals appear.
                self._force_fp32_until_safe = False
                self.pat.state = True
                self.pat.cnt = 0
                return self._lower_mode()
            if want_high and not want_low:
                # If we are seeing strong signals to remain in higher precision
                # stop treating the current fp32 mode as overflow-forced.
                self._force_fp32_until_safe = False
                return "fp32"
            # Otherwise keep the current mode (typically fp32) until safety
            # conditions are met.
            return current


        # 3) Apply hysteresis: if neither strongly wants change, keep current
        if want_low and not want_high:
            self.pat.update(True)
            target = self._lower_mode()
        elif want_high and not want_low:
            self.pat.update(False)
            target = self._promote_mode(current)
        else:
            # ambiguous; keep current
            return current

        # Enforce patience to avoid thrash
        if self.pat.state:
            return target
        return current


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
