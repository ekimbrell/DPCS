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
- with_activation_budget(budget_frac: Optional[float]) -> ContextManager

All helpers gracefully degrade on older PyTorch versions.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Mapping as _MappingABC, Sequence as _SequenceABC
from contextlib import contextmanager, nullcontext
from typing import Any, Callable, Deque, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple
import json
import math
import os
import time
import numbers

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


def grad_scaler_get_scale(scaler: Optional[torch.amp.GradScaler]) -> Optional[float]:
    """Return the current scale from ``GradScaler.get_scale`` when available."""

    if scaler is None:
        return None
    get_scale = getattr(scaler, "get_scale", None)
    if get_scale is None or not callable(get_scale):
        return None
    try:
        value = get_scale()
    except Exception:
        return None
    try:
        return float(value)
    except Exception:
        try:
            return float(value.item())  # type: ignore[union-attr]
        except Exception:
            return None


def grad_scaler_found_inf(scaler: Optional[torch.amp.GradScaler]) -> Optional[bool]:
    """Inspect ``GradScaler`` internal ``found_inf`` flags.

    Returns ``True`` when any optimizer recorded overflow in the current step,
    ``False`` when checks were performed and no overflow was observed, and
    ``None`` when the information is unavailable (older PyTorch or disabled
    scaler).
    """

    if scaler is None:
        return None
    per_opt = getattr(scaler, "_per_optimizer_states", None)
    seen = False

    if isinstance(per_opt, Mapping):
        try:
            states = list(per_opt.values())
        except Exception:
            states = []
        for state in states:
            if not isinstance(state, Mapping):
                continue
            found_map = state.get("found_inf_per_device")
            if not isinstance(found_map, Mapping):
                continue
            for found_inf in list(found_map.values()):
                seen = True
                try:
                    if float(found_inf) != 0.0:
                        return True
                except Exception:
                    try:
                        if bool(found_inf):
                            return True
                    except Exception:
                        continue
        if seen:
            return False
    return None


class AmpOverflowMonitor:
    """Context manager tracking ``GradScaler`` overflow observations.

    The monitor wraps :meth:`GradScaler.update` and records whether overflow was
    detected via ``found_inf`` flags or scale drops. A bounded history keeps the
    last ``history`` observations, enabling ``overflow_recent(K)`` queries.
    """

    def __init__(
        self,
        scaler: torch.amp.GradScaler,
        *,
        history: int = 4,
        on_overflow: Optional[Callable[[bool], None]] = None,
    ) -> None:
        if scaler is None:
            raise TypeError("scaler must be a GradScaler instance")
        self._scaler = scaler
        try:
            hist = int(history)
        except Exception:
            hist = 4
        self._history_len = max(1, hist)
        self._history: Deque[bool] = deque(maxlen=self._history_len)
        self._scale_before: Optional[float] = None
        self._found_inf_before: Optional[bool] = None
        self._last_overflow: bool = False
        self._last_scale: Optional[float] = grad_scaler_get_scale(scaler)
        self._active = False
        self._callback = on_overflow

    @property
    def scaler(self) -> torch.amp.GradScaler:
        return self._scaler

    @property
    def last_overflow(self) -> bool:
        return bool(self._last_overflow)

    @property
    def last_scale(self) -> Optional[float]:
        return self._last_scale

    @property
    def history(self) -> Tuple[bool, ...]:
        return tuple(self._history)

    def overflow_recent(self, steps: Optional[int] = None) -> bool:
        if not self._history:
            return bool(self._last_overflow) if steps is None or steps > 0 else False
        if steps is None:
            limit = len(self._history)
        else:
            try:
                limit = max(1, int(steps))
            except Exception:
                limit = len(self._history)
        count = 0
        for flag in reversed(self._history):
            if flag:
                return True
            count += 1
            if count >= limit:
                break
        return False

    def _record(self, overflow: bool) -> None:
        flag = bool(overflow)
        self._last_overflow = flag
        self._history.append(flag)
        if self._callback is not None:
            try:
                self._callback(flag)
            except Exception:
                pass

    def __enter__(self) -> torch.amp.GradScaler:
        if self._active:
            return self._scaler
        self._active = True
        self._scale_before = grad_scaler_get_scale(self._scaler)
        self._found_inf_before = grad_scaler_found_inf(self._scaler)
        return self._scaler

    def __exit__(self, exc_type, exc, tb) -> bool:
        scale_after = grad_scaler_get_scale(self._scaler)
        found_inf_after = grad_scaler_found_inf(self._scaler)
        overflow = False
        if self._found_inf_before is True or found_inf_after is True:
            overflow = True
        elif (
            self._scale_before is not None
            and scale_after is not None
            and scale_after < self._scale_before
        ):
            overflow = True
        self._scale_before = None
        self._found_inf_before = None
        self._active = False
        if scale_after is not None:
            self._last_scale = scale_after
        self._record(overflow)
        return False

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


def _coerce_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, numbers.Integral):
        return int(value)
    raise TypeError("expected an integer value")


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, numbers.Integral):
        return bool(value)
    raise TypeError("expected a boolean value")


def _coerce_float(value: Any) -> float:
    if isinstance(value, numbers.Real):
        return float(value)
    raise TypeError("expected a real value")


def _coerce_precision_mix(value: Any) -> Dict[str, int]:
    if not isinstance(value, _MappingABC):
        raise TypeError("precision_mix must be a mapping")
    result: Dict[str, int] = {}
    for key, count in value.items():
        result[str(key)] = _coerce_int(count)
    return result


def _coerce_sequence(value: Any, *, item_coerce: Callable[[Any], Any]) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError("sequence value must not be a string")
    if not isinstance(value, _SequenceABC):
        raise TypeError("expected a sequence value")
    return [item_coerce(item) for item in value]


def _coerce_break_reasons(value: Any) -> List[Dict[str, Any]]:
    reasons = _coerce_sequence(value, item_coerce=lambda x: x)
    normalized: List[Dict[str, Any]] = []
    for item in reasons:
        if isinstance(item, _MappingABC):
            reason = str(item.get("reason", ""))
            count = _coerce_int(item.get("count", 0))
        elif isinstance(item, (str, bytes, bytearray)):
            reason = str(item)
            count = 0
        elif isinstance(item, _SequenceABC):
            seq = list(item)
            reason = str(seq[0]) if seq else ""
            count = _coerce_int(seq[1]) if len(seq) > 1 else 0
        else:
            reason = str(item)
            count = 0
        normalized.append({"reason": reason, "count": count})
    return normalized


SchemaEntry = Dict[str, Any]


TELEMETRY_SCHEMA: Dict[str, SchemaEntry] = {
    "step": {"required": True, "coerce": _coerce_int},
    "broadcast_step": {"required": True, "coerce": _coerce_int},
    "headroom_frac": {"required": True, "coerce": _coerce_float, "allow_none": True},
    "amp_mode": {"required": True, "coerce": str},
    "overflow": {"required": True, "coerce": _coerce_bool},
    "overflow_flag": {"required": True, "coerce": _coerce_bool},
    "cooldown_remaining": {"required": True, "coerce": _coerce_int},
    "num_checkpointed": {"required": True, "coerce": _coerce_int},
    "ckpt_on": {"required": True, "coerce": _coerce_bool},
    "num_leaves": {"required": True, "coerce": _coerce_int},
    "step_peak_bytes": {"required": True, "coerce": _coerce_int},
    "peak_alloc_bytes": {"required": True, "coerce": _coerce_int},
    "allocated": {"required": True, "coerce": _coerce_int},
    "reserved": {"required": True, "coerce": _coerce_int},
    "active": {"required": True, "coerce": _coerce_int},
    "fragmentation_hint": {"required": True, "coerce": _coerce_float},
    "device_free": {
        "required": True,
        "coerce": _coerce_int,
        "allow_none": True,
    },
    "device_total": {
        "required": True,
        "coerce": _coerce_int,
        "allow_none": True,
    },
    "precision_mix": {"required": True, "coerce": _coerce_precision_mix},
    "free_bytes": {"required": False, "coerce": _coerce_int},
    "total_bytes": {"required": False, "coerce": _coerce_int},
    "grad_var_avg": {"required": False, "coerce": _coerce_float, "allow_none": True},
    "curv_avg": {"required": False, "coerce": _coerce_float, "allow_none": True},
    "ckpt_ids": {
        "required": False,
        "coerce": lambda value: _coerce_sequence(value, item_coerce=_coerce_int),
    },
    "ckpt_modules": {
        "required": False,
        "coerce": lambda value: _coerce_sequence(value, item_coerce=str),
    },
    "graph_breaks_total": {"required": False, "coerce": _coerce_int},
    "top_break_reasons": {
        "required": False,
        "coerce": _coerce_break_reasons,
    },
}


def _sanitize_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(record, _MappingABC):
        raise TypeError("record must be a mapping")
    sanitized: Dict[str, Any] = {}
    missing = [key for key, spec in TELEMETRY_SCHEMA.items() if spec.get("required") and key not in record]
    if missing:
        raise ValueError(f"missing telemetry fields: {', '.join(sorted(missing))}")
    unknown = set(record.keys()) - set(TELEMETRY_SCHEMA.keys())
    if unknown:
        raise ValueError(f"unknown telemetry fields: {', '.join(sorted(unknown))}")
    for key, value in record.items():
        spec = TELEMETRY_SCHEMA.get(key)
        if spec is None:
            raise ValueError(f"unknown telemetry field: {key}")
        if value is None:
            if spec.get("allow_none"):
                sanitized[key] = None
                continue
            raise TypeError(f"telemetry field '{key}' does not allow None")
        coerce = spec.get("coerce")
        if coerce is not None:
            try:
                coerced = coerce(value)
            except Exception as exc:
                raise TypeError(f"invalid value for telemetry field '{key}': {value!r}") from exc
        else:
            coerced = value
        sanitized[key] = coerced
    return sanitized


def allocator_telemetry(device: Optional[int | torch.device] = None) -> Dict[str, Any]:
    stats = {
        "allocated": 0,
        "reserved": 0,
        "active": 0,
        "fragmentation_hint": 0.0,
    }
    if not torch.cuda.is_available():
        return stats
    if device is None:
        try:
            index = torch.cuda.current_device()
        except Exception:
            index = 0
        candidate: Any = f"cuda:{index}"
    else:
        candidate = device
    try:
        dev = torch.device(candidate)
    except Exception:
        if device is not None:
            return stats
        try:
            dev = torch.device("cuda:0")
        except Exception:
            return stats

    def _call(name: str, fallback: Optional[str] = None) -> int:
        target = getattr(torch.cuda, "memory", torch.cuda)
        fn = getattr(target, name, None)
        if fn is None and fallback is not None:
            fn = getattr(torch.cuda, fallback, None)
        if fn is None:
            return 0
        try:
            value = fn(dev)
        except TypeError:
            try:
                value = fn()
            except Exception:
                return 0
        except Exception:
            return 0
        try:
            return int(value)
        except Exception:
            try:
                return int(value.item())  # type: ignore[union-attr]
            except Exception:
                return 0

    allocated = _call("memory_allocated", "memory_allocated")
    reserved = _call("memory_reserved", "memory_reserved")

    stats["allocated"] = allocated
    stats["reserved"] = reserved

    try:
        raw_stats = torch.cuda.memory_stats(dev)
    except TypeError:
        try:
            raw_stats = torch.cuda.memory_stats()
        except Exception:
            raw_stats = None
    except Exception:
        raw_stats = None

    if isinstance(raw_stats, _MappingABC):
        try:
            stats["active"] = int(raw_stats.get("active_bytes.all.current", allocated))
        except Exception:
            stats["active"] = allocated
        frag = raw_stats.get("fragmentation", raw_stats.get("fragmentation_metric"))
        if frag is None:
            try:
                inactive = int(raw_stats.get("inactive_split_bytes.all.current", 0))
                total = reserved if reserved > 0 else int(raw_stats.get("reserved_bytes.all.current", reserved))
                frag = (inactive / total) if total else 0.0
            except Exception:
                frag = 0.0
        try:
            stats["fragmentation_hint"] = float(frag)
        except Exception:
            stats["fragmentation_hint"] = 0.0
    else:
        stats["active"] = allocated
        stats["fragmentation_hint"] = 0.0

    return stats


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
        data = _sanitize_record(record)
        try:
            line = self._encode(data)
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

@contextmanager
def with_activation_budget(budget_frac: Optional[float]) -> Iterator[None]:
    """Temporarily enable PyTorch selective checkpointing activation budgets."""

    if budget_frac is None:
        yield
        return

    try:
        budget = float(budget_frac)
    except Exception:
        budget = 0.0
    if not math.isfinite(budget):
        budget = 0.0
    budget = max(0.0, min(1.0, budget))

    targets: List[Tuple[Any, str, Any]] = []

    def _prepare_target(obj: Any, attr: str) -> None:
        if obj is None or not hasattr(obj, attr):
            return
        try:
            prev = getattr(obj, attr)
        except Exception:
            prev = None
        try:
            setattr(obj, attr, budget)
        except Exception:
            return
        targets.append((obj, attr, prev))

    functorch_cfg = getattr(getattr(torch, "_functorch", None), "config", None)
    torch_cfg = getattr(torch, "config", None)
    for cfg in (functorch_cfg, torch_cfg):
        for name in (
            "activation_memory_budget",
            "selective_checkpointing_activation_memory_budget",
        ):
            _prepare_target(cfg, name)

    try:
        yield
    finally:
        for obj, attr, prev in reversed(targets):
            try:
                setattr(obj, attr, prev)
            except Exception:
                pass

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


_BROADCAST_STATE: Dict[int, Dict[str, Any]] = {}


def _broadcast_state_key(group: Optional[Any]) -> int:
    return id(group) if group is not None else -1


def dist_broadcast(obj: torch.Tensor, group: Optional[Any] = None) -> bool:
    """Broadcast ``obj`` from rank 0 with a strict participation contract.

    The helper coordinates a per-step rendezvous that first synchronizes all
    ranks via :func:`torch.distributed.barrier` followed by a lightweight
    ``all_reduce`` on a monotonically increasing step counter. Any mismatch in
    participation (e.g. a missing rank) triggers a descriptive ``RuntimeError``
    rather than hanging indefinitely.

    Parameters
    ----------
    obj:
        Tensor to broadcast. Rank 0 is treated as the source and must populate
        ``obj`` before calling.
    group:
        Optional process group to use. ``None`` defaults to the global group.

    Returns
    -------
    bool
        ``True`` when a distributed backend was available and the broadcast was
        performed. ``False`` when :mod:`torch.distributed` is unavailable.
    """

    dist = _distributed_backend()
    if dist is None:
        return False

    backend = str(dist.get_backend(group=group)).lower()
    world_size = int(dist.get_world_size(group=group))

    if world_size <= 1:
        dist.broadcast(obj, src=0, group=group)
        return True

    if backend == "nccl":
        if not obj.is_cuda:
            raise RuntimeError(
                "dist_broadcast requires CUDA tensors when using the NCCL backend"
            )
        device = obj.device
    else:
        device = torch.device("cpu")

    key = _broadcast_state_key(group)
    state = _BROADCAST_STATE.get(key)
    if state is None or state.get("device") != device:
        step_tensor = torch.zeros(1, dtype=torch.long, device=device)
        state = {"step": 0, "tensor": step_tensor, "device": device}
        _BROADCAST_STATE[key] = state

    step_tensor = state["tensor"]
    state["step"] = int(state.get("step", 0)) + 1
    step = state["step"]
    step_tensor.fill_(step)

    try:
        dist.barrier(group=group)
    except RuntimeError as exc:
        raise RuntimeError(
            f"DPCS dist_broadcast barrier failed at step {step}: "
            "ensure all ranks call dist_broadcast each step"
        ) from exc

    try:
        dist.all_reduce(step_tensor, group=group)
    except RuntimeError as exc:
        raise RuntimeError(
            f"DPCS dist_broadcast participation check failed at step {step}: "
            "all_reduce did not complete"
        ) from exc

    total = int(step_tensor.item())
    expected = step * world_size
    if total != expected:
        raise RuntimeError(
            "DPCS dist_broadcast participation check failed at step "
            f"{step}: expected sum {expected}, got {total}. "
            "Ensure every rank calls dist_broadcast once per scheduler step."
        )

    step_tensor.fill_(step)

    try:
        dist.broadcast(obj, src=0, group=group)
    except RuntimeError as exc:
        raise RuntimeError("DPCS dist_broadcast failed during broadcast") from exc

    return True


__all__ = [
    "amp_preferred_dtype",
    "amp_enabled",
    "amp_uses_grad_scaler",
    "grad_scaler_get_scale",
    "grad_scaler_found_inf",
    "AmpOverflowMonitor",
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
    "allocator_telemetry",
    "TELEMETRY_SCHEMA",
    "JsonlLogger",
    "dist_is_initialized",
    "dist_get_rank",
    "dist_world_size",
    "dist_broadcast",
    "checkpoint_call",
    "with_activation_budget",
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
