"""DPCS configuration.

This module defines a single frozen dataclass, ``DPCSConfig``, that holds all
scheduler knobs. It mirrors the CLI keys your runner forwards as ``--dpcs.*``
(e.g. ``--dpcs.curv_period``). Unknown or ``None`` values are ignored.

Typical use inside dpcs.py:

    from .config import DPCSConfig
    cfg = DPCSConfig.from_kwargs(**kwargs)

Keep this file *pure*: no imports from other project files. Avoid runtime
allocations in hot paths; the config is immutable and cheap to pass around.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Dict
import torch


def _default_device_type() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _clip01(x: float) -> float:
    # fast, branch-free-ish clip to [0, 1]
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


@dataclass(frozen=True)
class DPCSConfig:
    """Immutable configuration for the Dynamic Precision & Checkpointing Scheduler.

    Notes on types:
    - Flags coming from CLI may be ``int`` (0/1). We keep integer types for
      flags that are forwarded directly to low-level toggles to avoid surprises.
    - Fractions are clipped into [0, 1] during construction.
    """

    # -------------------------- Runtime / device --------------------------
    device_type: str = field(default_factory=_default_device_type)  # "cuda" | "cpu"
    enable_precision: int = 1  # 1 to allow precision changes; 0 to freeze at AMP choice

    # --------------------- Precision decision thresholds ------------------
    # Gradient-variance bands (EMA) for demotion/promotion
    epsilon_g_low: float = 1e-6
    epsilon_g_high: float = 1e-4
    # Curvature proxy (e.g., power-iter HVP top-1 eig estimate) bands
    kappa_low: float = 1e-3
    kappa_high: float = 1e-1
    # Hysteresis/patience to avoid mode thrash
    mode_patience: int = 16

    # ---------------- Curvature probe budget / scheduling -----------------
    curv_period: int = 50              # probe every N steps (0 disables)
    hvp_power_iters: int = 2           # iterations per probed leaf
    max_modules_per_probe: int = 1     # how many leaves to probe per probe-step

    # ----------------------- Memory / headroom gates ----------------------
    # Use free/total VRAM to gate precision & checkpointing decisions.
    low_headroom_frac: float = 0.08    # below this => aggressive saving
    hi_headroom_frac: float = 0.22     # above this => relax saving

    # --------------------- Activation checkpointing -----------------------
    ckpt_patience: int = 8             # steps before changing ckpt plan
    ckpt_topk_frac: float = 0.20       # fraction of heaviest leaves to ckpt
    ckpt_use_benefit_score: int = 1    # 1: bytes / fwd_ms_ema ranking; 0: bytes
    min_activation_bytes_to_ckpt: int = 1 << 20  # ignore tiny activations (<1 MiB)
    delegate_selective_ckpt: bool = False  # delegate to PyTorch selective ckpt
    activation_memory_budget_frac: float | None = None  # budget when delegating

    # --------------------- TransformerEngine FP8 (opt) --------------------
    te_amax_history_len: int = 16
    te_margin_init: int = 0
    te_margin_inc: int = 1
    te_margin_dec: int = 1

    # ----------------------------- Logging --------------------------------
    log_every: int = 1  # emit JSONL every step by default

    # ------------------------- Compile diagnostics ------------------------
    compile_diagnostics: bool = False
    compile_warmup_steps: int = 20
    no_flip_during_warmup: bool = True

    # ----------------------------- Helpers --------------------------------
    def amp_dtype(self) -> torch.dtype:
        """Return the autocast dtype preference for this device.

        Prefers bfloat16 on CUDA if supported; falls back to float16; otherwise
        returns float32 on CPU.
        """
        if self.device_type == "cuda" and torch.cuda.is_available():
            try:
                if torch.cuda.is_bf16_supported():
                    return torch.bfloat16
            except Exception:
                # Older PyTorch without is_bf16_supported
                pass
            return torch.float16
        return torch.float32

    # Frozen dataclass: use a constructor to normalize/validate inputs.
    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> "DPCSConfig":
        """Construct a config from possibly sparse/None-heavy kwargs.

        * Unknown keys are ignored.
        * ``None`` values are ignored (keep defaults).
        * Fractions are clipped into [0, 1].
        * Negative patience/periods are clipped to zero.
        """
        valid = {f.name for f in fields(cls)}
        init: Dict[str, Any] = {}
        for k, v in kwargs.items():
            if k not in valid or v is None:
                continue
            init[k] = v

        # Post-normalize select fields
        def _nz_int(name: str):
            if name in init:
                init[name] = max(0, int(init[name]))

        def _frac(name: str):
            if name in init:
                init[name] = float(_clip01(float(init[name])))

        _nz_int("mode_patience")
        _nz_int("curv_period")
        _nz_int("hvp_power_iters")
        _nz_int("max_modules_per_probe")
        _nz_int("ckpt_patience")
        _nz_int("min_activation_bytes_to_ckpt")
        _nz_int("te_amax_history_len")
        _nz_int("te_margin_init")
        _nz_int("te_margin_inc")
        _nz_int("te_margin_dec")
        _nz_int("log_every")
        _nz_int("compile_warmup_steps")

        _frac("low_headroom_frac")
        _frac("hi_headroom_frac")
        _frac("ckpt_topk_frac")
        _frac("activation_memory_budget_frac")

        for name in ("compile_diagnostics", "no_flip_during_warmup", "delegate_selective_ckpt"):
            if name in init:
                init[name] = bool(init[name])

        # Ensure bands are ordered (low <= high)
        if "epsilon_g_low" in init and "epsilon_g_high" in init:
            lo, hi = float(init["epsilon_g_low"]), float(init["epsilon_g_high"])
            if lo > hi:
                init["epsilon_g_low"], init["epsilon_g_high"] = hi, lo
        if "kappa_low" in init and "kappa_high" in init:
            lo, hi = float(init["kappa_low"]), float(init["kappa_high"])
            if lo > hi:
                init["kappa_low"], init["kappa_high"] = hi, lo

        return cls(**init)  # type: ignore[arg-type]

    def to_kwargs(self) -> Dict[str, Any]:
        """Return a plain dict suitable for logging or reconstruction."""
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass(frozen=True)
class TelemetryCfg:
    """Lightweight knobs for optional runtime telemetry hooks."""

    sample_every: int = 16
    sampled_modules_regex: str = "(attn|ffn|bottleneck)$"
    enable_timing: bool = False


@dataclass(frozen=True)
class CheckpointCfg:
    """Lightweight knobs for activation checkpointing wrappers."""

    preserve_rng_state: bool = True
    use_reentrant: bool = False
    max_fraction: float = 0.5

    def __post_init__(self) -> None:  # pragma: no cover - trivial setters
        object.__setattr__(self, "preserve_rng_state", bool(self.preserve_rng_state))
        object.__setattr__(self, "use_reentrant", bool(self.use_reentrant))
        object.__setattr__(self, "max_fraction", float(_clip01(float(self.max_fraction))))


__all__ = ["DPCSConfig", "TelemetryCfg", "CheckpointCfg"]
