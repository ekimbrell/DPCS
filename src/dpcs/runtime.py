"""Runtime utilities for DPCS (AMP, SDPA, memory, checkpoint).

This module intentionally contains *only* fast, well-scoped helpers used by the
scheduler and runner. Importing it must be cheap and side-effect free.

Exposed functions (stable):
- amp_preferred_dtype(device_type: str) -> torch.dtype
- amp_enabled(device_type: str) -> bool
- amp_uses_grad_scaler(dtype: torch.dtype, device_type: str) -> bool
- sdpa_backends_normalize(backends: Iterable[Any]) -> Tuple[Any, ...]
- sdpa_context(force: bool, backends: Iterable[Any]) -> ContextManager
- get_mem_info(device: Optional[int | torch.device]) -> tuple[int, int]
- reset_step_peak(device: Optional[int | torch.device]) -> None
- get_step_peak(device: Optional[int | torch.device]) -> int
- reset_peak_memory_stats() -> None
- max_memory_allocated() -> int
- mem_get_info() -> Optional[Tuple[int, int]]
- headroom_frac() -> Optional[float]
- checkpoint_call(fn: Callable, *args, determinism_check: str = "none", **kwargs) -> Any

All helpers gracefully degrade on older PyTorch versions.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple
import json
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

def get_mem_info(device: Optional[torch.device | int] = None) -> tuple[int, int]:
    """Return ``(free_bytes, total_bytes)`` from :func:`torch.cuda.memory.mem_get_info`."""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for get_mem_info")
    try:
        free, total = torch.cuda.memory.mem_get_info(device)
    except AttributeError:
        legacy = getattr(torch.cuda, "mem_get_info", None)
        if legacy is None:
            raise RuntimeError("torch.cuda.mem_get_info is unavailable")
        try:
            free, total = legacy(device) if device is not None else legacy()
        except TypeError:
            free, total = legacy()
    return int(free), int(total)


def reset_step_peak(device: Optional[torch.device | int] = None) -> None:
    """Reset allocator peaks via :func:`torch.cuda.memory.reset_peak_memory_stats`."""

    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.memory.reset_peak_memory_stats(device)
    except AttributeError:
        legacy = getattr(torch.cuda, "reset_peak_memory_stats", None)
        if legacy is None:
            return
        if device is None:
            legacy()
        else:
            try:
                legacy(device)
            except TypeError:
                legacy()


def get_step_peak(device: Optional[torch.device | int] = None) -> int:
    """Return step peak bytes from :func:`torch.cuda.memory.max_memory_allocated` since reset."""

    if not torch.cuda.is_available():
        return 0
    try:
        return int(torch.cuda.memory.max_memory_allocated(device))
    except AttributeError:
        legacy = getattr(torch.cuda, "max_memory_allocated", None)
        if legacy is None:
            return 0
        if device is None:
            return int(legacy())
        try:
            return int(legacy(device))
        except TypeError:
            return int(legacy())


def reset_peak_memory_stats() -> None:
    """Legacy alias for :func:`reset_step_peak`."""

    reset_step_peak()


def max_memory_allocated() -> int:
    """Legacy alias returning :func:`get_step_peak`."""

    return get_step_peak()


def mem_get_info() -> Optional[Tuple[int, int]]:
    """Legacy alias for :func:`get_mem_info` returning ``None`` when CUDA is absent."""

    try:
        free, total = get_mem_info()
    except RuntimeError:
        return None
    return (free, total)

def headroom_frac():
    """Return [0..1] free/total headroom.
    Prefer cudaMemGetInfo; fall back to allocator view if it's bogus.
    """
    if torch.cuda.is_available():
        # 1) Try global device info (cudaMemGetInfo)
        try:
            free, total = get_mem_info()
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

# Logging helper -------------------------------------------------------------


class JsonlLogger:
    """Buffered JSONL writer.

    Records are accumulated in-memory and written to disk in batches to avoid
    per-step filesystem jitter. ``flush_every`` controls the batch size.
    """

    def __init__(self, path: str, *, flush_every: int = 20) -> None:
        if not isinstance(path, str):
            path = os.fspath(path)
        self._path = path
        try:
            flush = int(flush_every)
        except Exception:
            flush = 20
        self._flush_every = max(1, flush)
        self._buffer: List[str] = []
        self._closed = False
        directory = os.path.dirname(self._path)
        if directory:
            try:
                os.makedirs(directory, exist_ok=True)
            except Exception:
                pass

    def _encode(self, record: Mapping[str, Any]) -> str:
        try:
            return json.dumps(dict(record))
        except TypeError:
            sanitized: Dict[str, Any] = {}
            for key, value in dict(record).items():
                try:
                    json.dumps({key: value})
                    sanitized[key] = value
                except TypeError:
                    sanitized[key] = repr(value)
            return json.dumps(sanitized)

    def log(self, record: Mapping[str, Any]) -> None:
        if self._closed:
            return
        if not isinstance(record, Mapping):
            return
        try:
            line = self._encode(record)
        except Exception:
            return
        self._buffer.append(line)
        if len(self._buffer) >= self._flush_every:
            self.flush()

    def flush(self) -> None:
        if self._closed or not self._buffer:
            return
        data = "\n".join(self._buffer) + "\n"
        try:
            with open(self._path, "a", encoding="utf-8") as fh:
                fh.write(data)
        except Exception:
            return
        self._buffer.clear()

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.flush()
        finally:
            self._closed = True

    def __del__(self) -> None:  # pragma: no cover - best effort cleanup
        try:
            self.close()
        except Exception:
            pass

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

def _distributed_backend():
    """Return ``torch.distributed`` when available and initialized."""

    dist = getattr(torch, "distributed", None)
    if dist is None:
        return None

    is_available = getattr(dist, "is_available", None)
    try:
        if not callable(is_available) or not is_available():
            return None
    except Exception:
        return None

    is_initialized = getattr(dist, "is_initialized", None)
    try:
        if not callable(is_initialized) or not is_initialized():
            return None
    except Exception:
        return None

    return dist


def dist_is_initialized() -> bool:
    """Return ``True`` when :mod:`torch.distributed` is ready for collectives."""

    return _distributed_backend() is not None


def dist_get_rank(default: int = 0) -> int:
    """Return the current rank when available, else ``default``."""

    dist = _distributed_backend()
    if dist is None:
        return int(default)

    get_rank = getattr(dist, "get_rank", None)
    if not callable(get_rank):
        return int(default)

    try:
        return int(get_rank())
    except Exception:
        return int(default)


def dist_world_size(default: int = 1) -> int:
    """Return world size when available, else ``default``."""

    dist = _distributed_backend()
    if dist is None:
        return int(default)

    get_world_size = getattr(dist, "get_world_size", None)
    if not callable(get_world_size):
        return int(default)

    try:
        return int(get_world_size())
    except Exception:
        return int(default)


def dist_broadcast(tensor: torch.Tensor, src: int = 0) -> bool:
    """Broadcast ``tensor`` from ``src`` rank when distributed is ready."""

    dist = _distributed_backend()
    if dist is None:
        return False

    try:
        dist.broadcast(tensor, src=int(src))
    except Exception:
        return False
    return True


__all__ = [
    "amp_preferred_dtype",
    "amp_enabled",
    "amp_uses_grad_scaler",
    "sdpa_backends_normalize",
    "sdpa_context",
    "te_prepare_fp8_modules",
    "te_fp8_autocast",
    "get_mem_info",
    "reset_step_peak",
    "get_step_peak",
    "reset_peak_memory_stats",
    "max_memory_allocated",
    "mem_get_info",
    "headroom_frac",
    "JsonlLogger",
    "dist_is_initialized",
    "dist_get_rank",
    "dist_world_size",
    "dist_broadcast",
    "checkpoint_call",
    "timed_cuda",
]


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA is not available; memory demo skipped.")
    else:
        dev = torch.cuda.current_device()
        reset_step_peak(dev)
        print(f"Initial peak bytes: {get_step_peak(dev)}")
        first = torch.empty((1024,), device=dev)
        print(f"Peak after first alloc: {get_step_peak(dev)}")
        del first
        reset_step_peak(dev)
        print(f"Peak after reset: {get_step_peak(dev)}")
        second = torch.empty((2048,), device=dev)
        print(f"Peak after second alloc: {get_step_peak(dev)}")
        del second
