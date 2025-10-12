"""DPCS scheduler (core runtime using modular policies/signals/runtime)

This module exposes the public class ``DPCS``. It orchestrates precision
selection and activation checkpointing by delegating to:
  - config.DPCSConfig (immutable knobs)
  - policies.PrecisionPolicy / CheckpointPolicy (pure decision logic)
  - signals.GradSignals / CurvatureSignals (lightweight signal collection)
  - runtime (AMP/SDPA/memory/ckpt utilities)

It keeps the Python hot path small and avoids per-step dynamic allocations.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import math
import os

import torch
import torch.nn as nn

from .config import DPCSConfig, CheckpointCfg
from .policies import PrecisionPolicy, CheckpointPolicy, PrecisionCfg
from .signals import GradSignals, CurvatureSignals, EMA, tensor_bytes
from .runtime import (
    AmpOverflowMonitor,
    amp_preferred_dtype,
    amp_uses_grad_scaler,
    reset_peak_memory_stats,
    max_memory_allocated,
    mem_get_info,
    headroom_frac,
    grad_scaler_get_scale,
    JsonlLogger,
    te_prepare_fp8_modules,
    te_fp8_autocast,
    dist_is_initialized,
    dist_get_rank,
    dist_broadcast,
)

# ------------------------ checkpointable leaf wrapper -----------------------

def _call_with_kwargs(fn: Callable[..., Any], kwargs: Dict[str, Any]) -> Callable[..., Any]:
    if not kwargs:
        return fn

    def _inner(*inputs: Any) -> Any:
        return fn(*inputs, **kwargs)

    return _inner


class _CkptWrap(nn.Module):
    """Wrap a leaf to optionally run under activation checkpointing."""

    def __init__(self, mod: nn.Module, ckpt_cfg: CheckpointCfg, name: Optional[str] = None) -> None:
        super().__init__()
        self.mod = mod
        self.use_ckpt: bool = False
        self._ckpt_cfg = ckpt_cfg
        self._leaf_name = name or type(mod).__qualname__
        self._act_bytes: int = 0
        self._act_ema = EMA(beta=0.9)
        self._fp8_ready: bool = False
        self._fp8_active: bool = False
        self._fp8_recipe: Optional[Any] = None
        self._autocast_enabled: bool = False
        self._autocast_dtype: torch.dtype = torch.float32
        self._autocast_device_type: str = "cuda"

    @property
    def leaf_name(self) -> str:
        return self._leaf_name

    @property
    def activation_bytes(self) -> int:
        return self._act_bytes

    @property
    def activation_bytes_ema(self) -> float:
        if self._act_ema.initialized:
            return float(self._act_ema.value)
        return float(self._act_bytes)

    def set_fp8_support(self, recipe: Optional[Any]) -> None:
        self._fp8_ready = recipe is not None
        self._fp8_recipe = recipe if self._fp8_ready else None
        if not self._fp8_ready:
            self._fp8_active = False

    def set_fp8_active(self, active: bool) -> None:
        self._fp8_active = bool(active and self._fp8_ready)

    def set_autocast(self, device_type: str, dtype: torch.dtype, enabled: bool) -> None:
        self._autocast_device_type = str(device_type)
        self._autocast_dtype = dtype
        self._autocast_enabled = bool(enabled)

    def forward(self, *args, **kwargs):  # type: ignore[override]
        ctx = te_fp8_autocast(self._fp8_active, self._fp8_recipe)
        amp_ctx = torch.autocast(
            device_type=self._autocast_device_type,
            dtype=self._autocast_dtype,
            enabled=self._autocast_enabled,
        )
        with ctx, amp_ctx:
            if self.use_ckpt:
                out = self._checkpoint_call(*args, **kwargs)
            else:
                out = self.mod(*args, **kwargs)
        self._act_bytes = tensor_bytes(out)
        self._act_ema.update(float(self._act_bytes))
        return out

    def _checkpoint_call(self, *args, **kwargs):
        try:
            from torch.utils.checkpoint import checkpoint as _ckpt
        except Exception:  # pragma: no cover - torch without checkpoint utility
            return self.mod(*args, **kwargs)

        fn = _call_with_kwargs(self.mod, dict(kwargs))
        try:
            return _ckpt(
                fn,
                *args,
                use_reentrant=bool(self._ckpt_cfg.use_reentrant),
                preserve_rng_state=bool(self._ckpt_cfg.preserve_rng_state),
            )
        except TypeError:
            return _ckpt(fn, *args)

    def __getattr__(self, name: str):
        mod = self.__dict__.get("mod", None)
        if mod is None:
            mod = self._modules.get("mod", None)
        if mod is not None:
            if name == "mod":
                return mod
            try:
                return getattr(mod, name)
            except AttributeError:
                pass
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")


# --------------------------- leaf discovery / wrap -------------------------

_LEAF_TYPES = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)

def _is_leaf(m: nn.Module) -> bool:
    return (len(list(m.children())) == 0) or isinstance(m, _LEAF_TYPES)


def _wrap_leaves(model: nn.Module, ckpt_cfg: CheckpointCfg) -> Tuple[nn.Module, List[_CkptWrap]]:
    wrappers: List[_CkptWrap] = []

    def _wrap(module: nn.Module, prefix: str) -> nn.Module:
        if _is_leaf(module):
            w = _CkptWrap(module, ckpt_cfg, prefix or type(module).__qualname__)
            wrappers.append(w)
            return w
        for name, child in list(module.named_children()):
            child_prefix = f"{prefix}.{name}" if prefix else name
            setattr(module, name, _wrap(child, child_prefix))
        return module

    model = _wrap(model, "")
    return model, wrappers


# --------------------------- compatibility views -----------------------------
class _LeafState:
    """Lightweight compatibility view of a wrapped module's state."""

    __slots__ = ("_sched", "index", "wrapper", "last_var_step", "gvar_ema", "last_curv_step")

    def __init__(self, sched: "DPCS", index: int, wrapper: _CkptWrap) -> None:
        self._sched = sched
        self.index = index
        self.wrapper = wrapper
        self.last_var_step: Optional[float] = None
        self.gvar_ema: Optional[float] = None
        self.last_curv_step: Optional[float] = None

    # --- precision ---------------------------------------------------------
    @property
    def mode(self) -> str:
        return self._sched._effective_amp_mode()

    @mode.setter
    def mode(self, value: str) -> None:
        self._sched.force_precision(str(value))

    @property
    def cool(self) -> int:
        pol = getattr(self._sched, "_prec_pol", None)
        return int(getattr(pol, "cooldown", 0)) if pol is not None else 0

    # --- checkpointing -----------------------------------------------------
    @property
    def ckpt_blacklisted(self) -> bool:
        return self._sched._compat_ckpt_blacklisted(self.index)


class _StateRegistry(Mapping[nn.Module, _LeafState]):
    """Mapping facade that accepts either wrappers or original modules as keys."""

    def __init__(self, states: Sequence[_LeafState]):
        self._states = list(states)
        self._by_wrapper: Dict[nn.Module, _LeafState] = {st.wrapper: st for st in self._states}
        self._by_orig: Dict[nn.Module, _LeafState] = {}
        self._keys: List[nn.Module] = []
        for st in self._states:
            orig = getattr(st.wrapper, "mod", None)
            if isinstance(orig, nn.Module):
                self._by_orig[orig] = st
                self._keys.append(orig)
            else:
                self._keys.append(st.wrapper)

    def __getitem__(self, key: nn.Module) -> _LeafState:
        st = self._by_wrapper.get(key)
        if st is None:
            st = self._by_orig.get(key)
        if st is None:
            raise KeyError(key)
        return st

    def __iter__(self):  # type: ignore[override]
        if self._keys:
            return iter(self._keys)
        return iter(self._by_wrapper.keys())

    def __len__(self) -> int:  # type: ignore[override]
        return len(self._by_wrapper)

    def __contains__(self, key: object) -> bool:
        return key in self._by_wrapper or key in self._by_orig

    def items(self):  # type: ignore[override]
        if self._keys:
            return [(k, self[k]) for k in self._keys]
        return list(self._by_wrapper.items())

    def values(self):  # type: ignore[override]
        return list(self._by_wrapper.values())

# --------------------------------- DPCS ------------------------------------

class DPCS:
    """Dynamic Precision & Checkpointing Scheduler (public API).

    Methods expected by the runner:
      - wrap(model) -> model
      - start_step()
      - collect_signals(loss, model=None)
      - end_step(optim, scaler=None)
      - get_amp_config() -> (device_type, dtype, enabled)
      - amp_uses_grad_scaler() -> bool
      - set_log_jsonl(path)

    Attributes referenced by the runner/tests:
      - cfg: DPCSConfig
      - _step: int
    """

    @staticmethod
    def _normalize_checkpoint_cfg(value: Any) -> CheckpointCfg:
        if value is None:
            return CheckpointCfg()
        if isinstance(value, CheckpointCfg):
            return value
        if isinstance(value, Mapping):
            return CheckpointCfg(**dict(value))
        raise TypeError("checkpoint_cfg must be a CheckpointCfg, mapping, or None")

    @staticmethod
    def _normalize_precision_cfg(value: Any) -> PrecisionCfg:
        if value is None:
            return PrecisionCfg()
        if isinstance(value, PrecisionCfg):
            return value
        if isinstance(value, Mapping):
            return PrecisionCfg(**dict(value))
        raise TypeError("precision_cfg must be a PrecisionCfg, mapping, or None")

    def __init__(self, **kwargs: Any) -> None:
        ckpt_cfg_in = kwargs.pop("checkpoint_cfg", None)
        prec_cfg_in = kwargs.pop("precision_cfg", None)
        self.cfg = DPCSConfig.from_kwargs(**kwargs)
        self._ckpt_cfg = self._normalize_checkpoint_cfg(ckpt_cfg_in)
        self._prec_cfg = self._normalize_precision_cfg(prec_cfg_in)
        self._step = 0

        self._precision_enabled = bool(self.cfg.enable_precision)
        self._manual_precision_mode: Optional[str] = None
        self._headroom_override: Optional[Callable[[], Optional[float]]] = None
        self._compat_states: List[_LeafState] = []
        self._registry: Mapping[nn.Module, _LeafState] = _StateRegistry([])
        self._leaf_names: List[str] = []
        self._module_index: Dict[str, int] = {}
        self._module_slots: List[int] = []
        self._module_names_sorted: List[str] = []
        self._decision_buffer: Optional[torch.Tensor] = None

        tick_every = getattr(self.cfg, "tick_every", 1)
        try:
            tick_i = max(1, int(tick_every))
        except Exception:
            tick_i = 1
        self._ddp_tick_every = tick_i

        # AMP mode selection (bf16 preferred when supported, otherwise fp16).
        self._amp_preferred_dtype = amp_preferred_dtype(self.cfg.device_type)
        self._amp_device_available = self.cfg.device_type == "cuda" and torch.cuda.is_available()
        self._amp_mode = "fp32"

        if not self._precision_enabled:
            self._manual_precision_mode = "fp32"
            self._amp_mode = "fp32"

        # FP8 runtime state
        self._fp8_supported = False
        self._fp8_wrappers: List[_CkptWrap] = []
        self._te_recipe: Optional[Any] = None
        self._te_margin = max(0, int(self.cfg.te_margin_init))
        self._te_margin_min = self._te_margin
        self._te_margin_inc = max(0, int(self.cfg.te_margin_inc))
        self._te_margin_dec = max(0, int(self.cfg.te_margin_dec))
        self._te_margin_max = max(self._te_margin_min + 32, self._te_margin_min + 4 * max(1, self._te_margin_inc))

        # Leaves and signals
        self._model: Optional[nn.Module] = None
        self._leaves: List[_CkptWrap] = []
        self._grads: Optional[GradSignals] = None
        self._curv: Optional[CurvatureSignals] = None
        self._ckpt_on: bool = True
        self._ckpt_selected: set[int] = set()

        # Policies
        bf16_supported = self._amp_device_available and self._amp_preferred_dtype is torch.bfloat16
        self._prec_pol = PrecisionPolicy(
            self._prec_cfg,
            bf16_supported=bf16_supported,
            amp_available=self._amp_device_available,
        )
        self._ckpt_pol = CheckpointPolicy(self.cfg, self._ckpt_cfg)

        # Overflow tracking
        self._last_scale: Optional[float] = None
        patience = max(1, int(self._prec_cfg.patience))
        self._overflow_recent_window = max(2, min(8, patience))
        self._overflow_monitors: Dict[int, AmpOverflowMonitor] = {}
        self._overflow_recent_flag: bool = False

        # Logging
        self._log_path: Optional[str] = None
        self._log_every = max(1, int(self.cfg.log_every))
        self._jsonl_logger: Optional[JsonlLogger] = None

        self._update_autocast()

    # ------------------------------ API ------------------------------------

    def wrap(self, model: nn.Module) -> nn.Module:
        model, leaves = _wrap_leaves(model, self._ckpt_cfg)
        self._model = model
        self._leaves = leaves
        self._leaf_names = [w.leaf_name for w in leaves]
        self._compat_states = [_LeafState(self, i, w) for i, w in enumerate(leaves)]
        self._registry = _StateRegistry(self._compat_states)
        self._decision_buffer = None
        self._rebuild_module_index()
        self._update_autocast()

        # Optional TransformerEngine swap for FP8
        self._fp8_supported = False
        self._fp8_wrappers = []
        self._te_recipe = None
        self._te_margin = self._te_margin_min
        if self.cfg.device_type == "cuda" and torch.cuda.is_available() and leaves:
            recipe, replacements = te_prepare_fp8_modules(
                (w.mod for w in leaves),
                margin=self.cfg.te_margin_init,
                amax_history_len=self.cfg.te_amax_history_len,
            )
            if recipe is not None and replacements:
                self._te_recipe = recipe
                self._fp8_supported = True
                try:
                    recipe.margin = int(self._te_margin)
                except Exception:
                    pass
                for w in leaves:
                    orig = w.mod
                    repl = replacements.get(id(orig))
                    if repl is not None:
                        repl.train(orig.training)
                        w.mod = repl
                        w.set_fp8_support(self._te_recipe)
                    else:
                        w.set_fp8_support(None)
                self._fp8_wrappers = [w for w in leaves if w._fp8_ready]
            else:
                for w in leaves:
                    w.set_fp8_support(None)
        else:
            for w in leaves:
                w.set_fp8_support(None)
        for w in leaves:
            w.set_fp8_active(False)
        self._prec_pol.update_fp8_support(
            self._fp8_supported,
            low_headroom=self.cfg.low_headroom_frac,
            high_headroom=self.cfg.hi_headroom_frac,
        )

        # Attach gradient hooks (per-parameter) aggregated per leaf
        self._grads = GradSignals(leaves, sample_max_elems=4096, beta=0.9)
        self._grads.attach()

        # Curvature probes (optional, disabled if period=0)
        self._curv = CurvatureSignals(
            leaves,
            curv_period=self.cfg.curv_period,
            hvp_power_iters=self.cfg.hvp_power_iters,
            max_modules_per_probe=self.cfg.max_modules_per_probe,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        self._refresh_compat_stats()
        return model

    def set_log_jsonl(self, path: str) -> None:
        path_str = str(path)
        self._log_path = path_str
        if self._jsonl_logger is not None:
            try:
                self._jsonl_logger.close()
            except Exception:
                pass
            self._jsonl_logger = None
        if not path_str:
            return
        directory = os.path.dirname(path_str)
        if directory:
            try:
                os.makedirs(directory, exist_ok=True)
            except Exception:
                pass
        self._jsonl_logger = JsonlLogger(path_str, flush_every=20)

    def get_amp_config(self) -> Tuple[str, torch.dtype, bool]:
        device_type = self.cfg.device_type
        if device_type != "cuda" or not torch.cuda.is_available():
            return (device_type, torch.float32, False)
        mode = self._effective_amp_mode()
        if mode == "fp8":
            dtype = self._amp_preferred_dtype
            if dtype not in (torch.bfloat16, torch.float16):
                dtype = torch.float16
            return ("cuda", dtype, True)
        if mode == "bf16" and self._amp_preferred_dtype is torch.bfloat16:
            return ("cuda", torch.bfloat16, True)
        if mode == "fp16":
            return ("cuda", torch.float16, True)
        return ("cuda", torch.float32, False)

    def amp_uses_grad_scaler(self) -> bool:
        dev, dtype, enabled = self.get_amp_config()
        return enabled and amp_uses_grad_scaler(dtype, dev)

    def amp_dtype(self) -> torch.dtype:
        _, dtype, _ = self.get_amp_config()
        return dtype

    def amp_overflow_recent(self, steps: Optional[int] = None) -> bool:
        if steps is None:
            limit = self._overflow_recent_window
        else:
            try:
                limit = max(1, int(steps))
            except Exception:
                limit = self._overflow_recent_window
        for monitor in list(self._overflow_monitors.values()):
            if monitor.overflow_recent(limit):
                return True
        return bool(self._overflow_recent_flag) if limit > 0 else False

    def precision_mix(self) -> Dict[str, int]:
        mode = self._effective_amp_mode()
        total = len(self._compat_states) if self._compat_states else len(self._leaves)
        if total <= 0:
            return {}
        return {mode: total}

    # ------------------------- compatibility helpers ----------------------

    def _update_autocast(self) -> None:
        device_type = self.cfg.device_type
        enabled = False
        dtype = torch.float32
        if device_type == "cuda" and torch.cuda.is_available():
            if self._amp_mode == "bf16":
                enabled = True
                dtype = torch.bfloat16
            elif self._amp_mode == "fp16":
                enabled = True
                dtype = torch.float16
        for w in self._leaves:
            w.set_autocast(device_type=device_type, dtype=dtype, enabled=enabled)

    def _effective_amp_mode(self) -> str:
        if self._manual_precision_mode is not None:
            return self._manual_precision_mode
        if not self._precision_enabled:
            return "fp32"
        return self._amp_mode

    def force_precision(self, mode: Optional[str]) -> None:
        if mode is None:
            self.clear_precision_override()
            return
        norm = str(mode).lower()
        if norm in {"auto", "none"}:
            self.clear_precision_override()
            return
        if norm not in {"fp32", "fp16", "bf16", "fp8"}:
            raise ValueError(f"Unsupported precision override: {mode}")
        self._manual_precision_mode = norm
        self._amp_mode = norm
        self._update_autocast()

    def force_fp32(self) -> None:
        self.force_precision("fp32")

    def clear_precision_override(self) -> None:
        self._manual_precision_mode = None
        if not self._precision_enabled:
            self._manual_precision_mode = "fp32"
            self._amp_mode = "fp32"
        self._update_autocast()

    def precision_override(self) -> Optional[str]:
        return self._manual_precision_mode

    def _refresh_compat_stats(self) -> None:
        if not self._compat_states:
            return
        if self._grads is not None:
            for st in self._compat_states:
                st.last_var_step = self._grads.last_var(st.index)
                st.gvar_ema = self._grads.grad_var(st.index)
        else:
            for st in self._compat_states:
                st.last_var_step = None
                st.gvar_ema = None

        if self._curv is not None:
            kappa = self._curv.kappa
            for st in self._compat_states:
                if st.index < len(kappa):
                    st.last_curv_step = kappa[st.index]
                else:
                    st.last_curv_step = None
        else:
            for st in self._compat_states:
                st.last_curv_step = None

    def _compat_ckpt_blacklisted(self, index: int) -> bool:
        pol = getattr(self, "_ckpt_pol", None)
        if pol is None:
            return False
        blacklist = getattr(pol, "_blacklist", None)
        if isinstance(blacklist, dict):
            return bool(blacklist.get(int(index), 0))
        return False

    # ------------------------- distributed helpers ------------------------

    def _rebuild_module_index(self) -> None:
        names = list(self._leaf_names)
        count = len(names)
        if count == 0:
            self._module_index = {}
            self._module_slots = []
            self._module_names_sorted = []
            return

        order = sorted(enumerate(names), key=lambda kv: (kv[1], kv[0]))
        self._module_index = {name: slot for slot, (_idx, name) in enumerate(order)}
        self._module_slots = [0] * count
        self._module_names_sorted = [name for _idx, name in order]
        for slot, (leaf_idx, _name) in enumerate(order):
            if 0 <= leaf_idx < count:
                self._module_slots[leaf_idx] = slot

    @staticmethod
    def _precision_to_bits(mode: str) -> int:
        mapping = {"fp32": 0, "bf16": 1, "fp16": 2, "fp8": 3}
        return mapping.get(str(mode).lower(), 0)

    @staticmethod
    def _bits_to_precision(bits: int) -> str:
        mapping = {0: "fp32", 1: "bf16", 2: "fp16", 3: "fp8"}
        return mapping.get(bits & 0b11, "fp32")

    def _ensure_decision_buffer(self, count: int) -> torch.Tensor:
        if count <= 0:
            count = 1
        device = torch.device("cpu")
        if self.cfg.device_type == "cuda" and torch.cuda.is_available():
            try:
                device = torch.device(torch.cuda.current_device())
            except Exception:
                device = torch.device("cuda")
        if (
            self._decision_buffer is None
            or self._decision_buffer.device != device
            or self._decision_buffer.numel() != count
        ):
            try:
                self._decision_buffer = torch.empty(count, dtype=torch.uint8, device=device)
            except Exception:
                self._decision_buffer = torch.empty(count, dtype=torch.uint8, device=torch.device("cpu"))
        return self._decision_buffer

    def _apply_decision_tensor(self, tensor: torch.Tensor) -> None:
        if tensor.numel() == 0:
            return
        try:
            base_val = int(tensor.view(-1)[0].item())
        except Exception:
            return

        self._amp_mode = self._bits_to_precision(base_val & 0b11)
        self._update_autocast()

        if not self._module_slots or not self._leaves:
            self._ckpt_selected = set()
            return

        selected: set[int] = set()
        for leaf_idx, slot in enumerate(self._module_slots):
            if slot >= tensor.numel():
                continue
            try:
                encoded = int(tensor[slot].item())
            except Exception:
                continue
            use_ckpt = bool(encoded & 0b100)
            self._leaves[leaf_idx].use_ckpt = use_ckpt
            if use_ckpt:
                selected.add(leaf_idx)
        self._ckpt_selected = selected

    def _broadcast_decisions(self) -> None:
        if not dist_is_initialized():
            return

        count = max(1, len(self._module_slots))
        tensor = self._ensure_decision_buffer(count)
        rank = dist_get_rank(default=0)
        base = self._precision_to_bits(self._amp_mode) & 0b11

        with torch.no_grad():
            if rank == 0:
                tensor.fill_(base)
                if self._module_slots:
                    for leaf_idx, slot in enumerate(self._module_slots):
                        value = base
                        if leaf_idx < len(self._leaves) and self._leaves[leaf_idx].use_ckpt:
                            value |= 0b100
                        tensor[slot] = value
                else:
                    tensor[0] = base
            else:
                tensor.zero_()

        if not dist_broadcast(tensor, src=0):
            return

        self._apply_decision_tensor(tensor)

    def _read_headroom(self) -> Optional[float]:
        if self._headroom_override is not None:
            try:
                return self._headroom_override()
            except Exception:
                return None
        return headroom_frac()

    @property
    def vram_headroom(self) -> Callable[[], Optional[float]]:
        return self._headroom_override or headroom_frac

    @vram_headroom.setter
    def vram_headroom(self, fn: Optional[Callable[[], Optional[float]]]) -> None:
        if fn is None:
            self._headroom_override = None
        elif callable(fn):
            self._headroom_override = fn
        else:
            raise TypeError("vram_headroom override must be callable or None")

    def enable_checkpointing(self, enabled: bool) -> None:
        self._ckpt_on = bool(enabled)

    def overflow_monitor(self, scaler: torch.amp.GradScaler) -> AmpOverflowMonitor:
        if scaler is None:
            raise TypeError("scaler must be a GradScaler instance")
        monitor = self._overflow_monitors.get(id(scaler))
        if monitor is None or monitor.scaler is not scaler:
            monitor = AmpOverflowMonitor(scaler, history=self._overflow_recent_window)
            self._overflow_monitors[id(scaler)] = monitor
        return monitor

    def start_step(self) -> None:
        # Step-scoped peak VRAM reset
        reset_peak_memory_stats()
        # Decay blacklist counters for ckpt policy
        self._ckpt_pol.tick()

        # Sync TransformerEngine recipe margin and toggle FP8 activation
        if self._te_recipe is not None:
            try:
                self._te_recipe.margin = int(self._te_margin)
            except Exception:
                pass
        fp8_active = self._fp8_supported and self._amp_mode == "fp8"
        for w in self._fp8_wrappers:
            w.set_fp8_active(fp8_active)

    def collect_signals(self, loss: Optional[torch.Tensor], model: Optional[nn.Module] = None) -> None:
        # Optional curvature probe on schedule (requires loss with graph)
        if loss is not None and self._curv is not None:
            try:
                self._curv.maybe_probe(loss)
            except RuntimeError:
                # Allow training to continue if the current step cannot build HVP
                pass
        self._refresh_compat_stats()

    def _sync_min_headroom(self, headroom: Optional[float]) -> Optional[float]:
        """Synchronize headroom across ranks via distributed MIN reduce."""
        if headroom is None or self.cfg.device_type != "cuda":
            return headroom

        dist = getattr(torch, "distributed", None)
        if dist is None:
            return headroom

        is_available = getattr(dist, "is_available", None)
        try:
            if not callable(is_available) or not is_available():
                return headroom
        except Exception:
            return headroom

        is_initialized = getattr(dist, "is_initialized", None)
        try:
            if not callable(is_initialized) or not is_initialized():
                return headroom
        except Exception:
            return headroom

        reduce_op = getattr(dist, "ReduceOp", None)
        if reduce_op is None or not hasattr(reduce_op, "MIN"):
            return headroom

        try:
            device = torch.device(self.cfg.device_type)
        except Exception:
            device = torch.device("cpu")

        try:
            tensor = torch.tensor([float(headroom)], device=device)
        except Exception:
            tensor = torch.tensor([float(headroom)], device=torch.device("cpu"))

        try:
            dist.all_reduce(tensor, op=reduce_op.MIN)
        except Exception:
            return headroom

        try:
            return float(tensor.item())
        except Exception:
            return headroom

    def end_step(self, optim: torch.optim.Optimizer, scaler: Optional[torch.amp.GradScaler] = None) -> None:
        # 1) Read memory headroom
        hr = self._read_headroom()
        hr = self._sync_min_headroom(hr)
        free_b: Optional[int] = None
        total_b: Optional[int] = None
        info = mem_get_info()

        if info is not None:
            free_b, total_b = info
        peak_b = max_memory_allocated()

        # 2) Detect AMP overflow via GradScaler signals
        overflow = False
        overflow_recent = False
        monitor = None
        if scaler is not None:
            monitor = self._overflow_monitors.get(id(scaler))
        if monitor is not None:
            overflow = monitor.last_overflow
            overflow_recent = monitor.overflow_recent(self._overflow_recent_window)
            last_scale = monitor.last_scale
            if last_scale is not None:
                self._last_scale = last_scale

        scaler_enabled = False
        curr_scale: Optional[float] = None
        if scaler is not None:
            is_enabled = getattr(scaler, "is_enabled", None)
            if callable(is_enabled):
                try:
                    scaler_enabled = bool(is_enabled())
                except Exception:
                    scaler_enabled = False
            curr_scale = grad_scaler_get_scale(scaler)

        if scaler_enabled and curr_scale is not None:
            if self._last_scale is not None and curr_scale < self._last_scale:
                overflow = True
            self._last_scale = curr_scale
        elif scaler is None:
            self._last_scale = None
        elif curr_scale is not None:
            self._last_scale = curr_scale
        elif scaler is not None and not scaler_enabled:
            self._last_scale = None

        if overflow:
            overflow_recent = True
        self._overflow_recent_flag = overflow_recent

        # 3) Precision policy (global)
        gvar = self._grads.grad_var_avg() if self._grads is not None else None
        curv_avg = None
        if self._curv is not None and self._curv.kappa:
            vals = [x for x in self._curv.kappa if x is not None]
            if vals:
                curv_avg = float(sum(vals) / len(vals))
        prev_mode = self._amp_mode
        override_mode = self._manual_precision_mode
        if not self._precision_enabled:
            override_mode = override_mode or "fp32"
        if override_mode is not None:
            new_mode = override_mode
        else:
            new_mode = self._prec_pol.decide(hr, gvar, curv_avg, overflow, prev_mode)
            if new_mode == "fp8" and not self._fp8_supported:
                if self._amp_preferred_dtype is torch.bfloat16:
                    new_mode = "bf16"
                elif self._amp_preferred_dtype is torch.float16:
                    new_mode = "fp16"
                else:
                    new_mode = "fp32"
        self._amp_mode = new_mode
        self._update_autocast()
        self._update_te_margin(prev_mode, new_mode, overflow)

        # 4) Checkpoint policy
        if self._leaves:
            if self._ckpt_on:
                act = [w.activation_bytes_ema for w in self._leaves]
                enable_ids = self._ckpt_pol.plan(act, hr, free_b, peak_b)
                self._ckpt_selected = set(enable_ids)
                for i, w in enumerate(self._leaves):
                    w.use_ckpt = (i in self._ckpt_selected)
            else:
                self._ckpt_selected.clear()
                for w in self._leaves:
                    w.use_ckpt = False

        if (self._step % self._ddp_tick_every) == 0:
            self._broadcast_decisions()

        # 5) Logging (optional)
        if self._jsonl_logger and (self._step % self._log_every == 0):
            num_ckpt = int(sum(1 for w in self._leaves if w.use_ckpt))
            precision_hist = self.precision_mix()
            rec = {
                "step": int(self._step),
                "headroom_frac": float(hr) if hr is not None else None,
                "amp_mode": self._amp_mode,
                "overflow": bool(overflow),
                "num_checkpointed": num_ckpt,
                "ckpt_on": num_ckpt,
                "num_leaves": int(len(self._leaves)),
                "step_peak_bytes": int(peak_b),
                "peak_alloc_bytes": int(peak_b),
                "free_bytes": int(free_b) if free_b is not None else 0,
                "total_bytes": int(total_b) if total_b is not None else 0,
                "precision_mix": precision_hist,
                "grad_var_avg": float(gvar) if gvar is not None else None,
                "curv_avg": float(curv_avg) if curv_avg is not None else None,
            }
            if self._ckpt_selected:
                ids = sorted(int(i) for i in self._ckpt_selected)
            else:
                ids = []
            rec["ckpt_ids"] = ids
            if ids and self._leaf_names:
                rec["ckpt_modules"] = [
                    self._leaf_names[i] if 0 <= i < len(self._leaf_names) else str(i)
                    for i in ids
                ]
            try:
                self._jsonl_logger.log(rec)
            except Exception:
                pass

        self._step += 1


    def _update_te_margin(self, prev_mode: str, new_mode: str, overflow: bool) -> None:
        if not self._fp8_supported or self._te_recipe is None:
            self._te_margin = self._te_margin_min
            return

        margin = int(self._te_margin)
        changed = False

        if prev_mode == "fp8" and overflow:
            if self._te_margin_inc > 0:
                margin = min(self._te_margin_max, margin + self._te_margin_inc)
                changed = margin != self._te_margin
        elif new_mode == "fp8":
            if prev_mode == "fp8" and self._te_margin_dec > 0 and margin > self._te_margin_min:
                margin = max(self._te_margin_min, margin - self._te_margin_dec)
                changed = margin != self._te_margin
        else:
            if self._te_margin_dec > 0 and margin > self._te_margin_min:
                margin = max(self._te_margin_min, margin - self._te_margin_dec)
                changed = margin != self._te_margin

        if changed:
            self._te_margin = int(margin)
            try:
                self._te_recipe.margin = int(self._te_margin)
            except Exception:
                pass


__all__ = ["DPCS", "AmpOverflowMonitor"]
