"""Runtime utilities for DPCS (AMP, SDPA, memory, checkpoint).

This module intentionally contains *only* fast, well-scoped helpers used by the
scheduler and runner. Importing it must be cheap and side-effect free.

Exposed functions (stable):
- amp_preferred_dtype(device_type: str) -> torch.dtype
- amp_enabled(device_type: str) -> bool
- amp_uses_grad_scaler(dtype: torch.dtype, device_type: str) -> bool
- sdpa_backends_normalize(backends: Iterable[Any]) -> Tuple[Any, ...]
- sdpa_context(force: bool, backends: Iterable[Any]) -> ContextManager
- reset_peak_memory_stats() -> None
- max_memory_allocated() -> int
- mem_get_info() -> Optional[Tuple[int, int]]
- headroom_frac() -> Optional[float]
- checkpoint_call(fn: Callable, *args, determinism_check: str = "none", **kwargs) -> Any

All helpers gracefully degrade on older PyTorch versions.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from typing import Any, Dict, Iterable, Optional, Tuple
import os
import time

import torch
import torch.nn as nn


# Optional imports ------------------------------------------------------------
try:
    from torch.nn.attention import sdpa_kernel, SDPBackend  # PyTorch >= 2.1
    _HAVE_SDPA = True
except Exception:  # pragma: no cover
    _HAVE_SDPA = False
    @contextmanager
    def sdpa_kernel(*_args, **_kwargs):  # type: ignore[misc]
        yield
    class SDPBackend:  # type: ignore[override]
        MATH = "MATH"; EFFICIENT_ATTENTION = "EFFICIENT_ATTENTION"; FLASH_ATTENTION = "FLASH_ATTENTION"

try:  # pragma: no cover - optional dependency
    import transformer_engine.pytorch as _te
    from transformer_engine.common import recipe as _te_recipe
    from transformer_engine.pytorch.fp8 import FP8GlobalStateManager as _FP8GSM
    _HAVE_TE = True
except Exception:  # pragma: no cover
    _te = None  # type: ignore[assignment]
    _te_recipe = None  # type: ignore[assignment]
    _FP8GSM = None  # type: ignore[assignment]
    _HAVE_TE = False

# AMP helpers -----------------------------------------------------------------

def _is_cuda(device_type: str) -> bool:
    return (device_type == "cuda") and torch.cuda.is_available()


def amp_preferred_dtype(device_type: str = "cuda") -> torch.dtype:
    """Prefer bfloat16 on CUDA (if supported), else float16, else float32.
    See: torch.amp autocast docs.
    """
    if _is_cuda(device_type):
        try:
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
        except Exception:
            pass
        return torch.float16
    return torch.float32


def amp_enabled(device_type: str = "cuda") -> bool:
    """Whether autocast should be enabled for the given device."""
    return _is_cuda(device_type)


def amp_uses_grad_scaler(dtype: torch.dtype, device_type: str = "cuda") -> bool:
    """Use GradScaler only for fp16 on CUDA.
    bf16 generally does *not* require loss scaling; fp32 doesn't use AMP.
    """
    return _is_cuda(device_type) and (dtype is torch.float16)

# SDPA helpers ----------------------------------------------------------------

def sdpa_backends_normalize(backends: Iterable[Any]) -> Tuple[Any, ...]:
    """Normalize a list of backends to SDPBackend enums when available."""
    norm = []
    for b in backends:
        if _HAVE_SDPA and isinstance(b, SDPBackend):
            norm.append(b)
        elif _HAVE_SDPA and isinstance(b, str):
            name = b.upper()
            if hasattr(SDPBackend, name):
                norm.append(getattr(SDPBackend, name))
        else:
            norm.append(b)
    return tuple(norm)


def sdpa_context(force: bool, backends: Iterable[Any]):
    """Return a context manager that pins SDPA to specific backends.

    If ``force`` is False, returns a no-op context. Otherwise, yields the
    sdpa_kernel([...]) context selecting the requested backends.
    """
    if not force:
        return nullcontext()
    return sdpa_kernel(sdpa_backends_normalize(backends))

# TransformerEngine FP8 helpers ----------------------------------------------

def _te_fp8_supported() -> bool:
    if not (_HAVE_TE and torch.cuda.is_available() and _FP8GSM is not None):
        return False
    try:
        ok, _ = _FP8GSM.is_fp8_available()
        return bool(ok)
    except Exception:
        return False


def _te_make_recipe(margin: int, amax_history_len: int):
    if not (_HAVE_TE and _te_recipe is not None):
        return None
    try:
        margin_i = max(0, int(margin))
        hist_i = max(1, int(amax_history_len))
        return _te_recipe.DelayedScaling(margin=margin_i, amax_history_len=hist_i)
    except Exception:
        return None


def _te_linear_from_torch(module: nn.Linear) -> Optional[nn.Module]:
    if not _te_fp8_supported():
        return None
    try:
        weight = module.weight
    except AttributeError:
        return None
    if weight is None or not isinstance(weight, torch.Tensor):
        return None
    device = weight.device
    if device.type != "cuda":
        return None
    bias = getattr(module, "bias", None)
    try:
        te_linear = _te.Linear(
            module.in_features,
            module.out_features,
            bias=(bias is not None),
            params_dtype=weight.dtype,
            device=device,
        )
    except Exception:
        return None
    try:
        state = module.state_dict()
        te_linear.load_state_dict(state, strict=False)
    except Exception:
        with torch.no_grad():
            te_linear.weight.copy_(weight)
            if bias is not None and hasattr(te_linear, "bias"):
                te_linear.bias.copy_(bias)
    te_linear.train(module.training)
    return te_linear


def _te_convert_module(module: nn.Module) -> Optional[nn.Module]:
    if isinstance(module, nn.Linear):
        return _te_linear_from_torch(module)
    return None


def te_prepare_fp8_modules(
    modules: Iterable[nn.Module],
    *,
    margin: int,
    amax_history_len: int,
) -> Tuple[Optional[Any], Dict[int, nn.Module]]:
    """Return (recipe, replacements) for modules convertible to TransformerEngine."""

    modules = list(modules)
    if not modules:
        return (None, {})
    recipe = _te_make_recipe(margin, amax_history_len)
    replacements: Dict[int, nn.Module] = {}
    if recipe is None:
        return (None, replacements)
    for mod in modules:
        new_mod = _te_convert_module(mod)
        if new_mod is not None:
            replacements[id(mod)] = new_mod
    if not replacements:
        return (None, {})
    return (recipe, replacements)


def te_fp8_autocast(enabled: bool, recipe: Optional[Any] = None):
    if not (_HAVE_TE and enabled):
        return nullcontext()
    kwargs = {"enabled": True}
    if recipe is not None:
        kwargs["fp8_recipe"] = recipe
    try:
        return _te.fp8_autocast(**kwargs)  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover
        return nullcontext()

# Memory helpers --------------------------------------------------------------

def reset_peak_memory_stats() -> None:
    """Reset CUDA peak memory stats if available (step-scoped peaks)."""
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.memory.reset_peak_memory_stats()
    except Exception:
        try:
            torch.cuda.reset_peak_memory_stats()  # older API
        except Exception:
            pass


def max_memory_allocated() -> int:
    """Return step-scoped peak allocated bytes if available, else 0."""
    if not torch.cuda.is_available():
        return 0
    try:
        return int(torch.cuda.memory.max_memory_allocated())
    except Exception:
        try:
            return int(torch.cuda.max_memory_allocated())
        except Exception:
            return 0


def mem_get_info() -> Optional[Tuple[int, int]]:
    """Return (free_bytes, total_bytes) for current CUDA device, or None on CPU."""
    if not torch.cuda.is_available():
        return None
    try:
        free, total = torch.cuda.memory.mem_get_info()
        return int(free), int(total)
    except Exception:
        try:
            free, total = torch.cuda.mem_get_info()
            return int(free), int(total)
        except Exception:
            return None

def headroom_frac():
    """Return [0..1] free/total headroom.
    Prefer cudaMemGetInfo; fall back to allocator view if it's bogus.
    """
    if torch.cuda.is_available():
        # 1) Try global device info (cudaMemGetInfo)
        try:
            free, total = torch.cuda.memory.mem_get_info()
            if free > 0 and total > 0:
                return free / total
        except Exception:
            pass
        # 2) Fallback: allocator vs physical total
        dev = torch.cuda.current_device()
        total_phys = torch.cuda.get_device_properties(dev).total_memory
        reserved = torch.cuda.memory.memory_reserved(dev)
        free_like = max(total_phys - reserved, 0)
        return free_like / max(total_phys, 1)
    return None

# Checkpoint helper -----------------------------------------------------------

def checkpoint_call(fn, *args, determinism_check: str = "none", **kwargs):
    """Call ``torch.utils.checkpoint.checkpoint`` with modern defaults.

    - Non-reentrant (use_reentrant=False) is recommended in recent PyTorch.
    - ``determinism_check``: "default" or "none" (passthrough to PyTorch)
    - We avoid stashing RNG state to reduce memory cost.
    """
    from torch.utils.checkpoint import checkpoint as _ckpt
    try:
        return _ckpt(fn, *args,
                     use_reentrant=False,
                     determinism_check=determinism_check,
                     preserve_rng_state=False,
                     **kwargs)
    except TypeError:
        # Older PyTorch without determinism_check / preserve_rng_state kwargs
        return _ckpt(fn, *args, use_reentrant=False, **kwargs)

# Convenience -----------------------------------------------------------------

@contextmanager
def timed_cuda() -> float:
    """Context manager that measures elapsed ms with CUDA sync if available.

    Usage:
        with timed_cuda() as tms:
            ...
        print(tms[0])  # elapsed milliseconds (float)
    """
    t = [0.0]
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        yield t
    finally:
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t[0] = (time.perf_counter() - t0) * 1e3
        except Exception:
            t[0] = 0.0

__all__ = [
    "amp_preferred_dtype",
    "amp_enabled",
    "amp_uses_grad_scaler",
    "sdpa_backends_normalize",
    "sdpa_context",
    "te_prepare_fp8_modules",
    "te_fp8_autocast",
    "reset_peak_memory_stats",
    "max_memory_allocated",
    "mem_get_info",
    "headroom_frac",
    "checkpoint_call",
    "timed_cuda",
]
