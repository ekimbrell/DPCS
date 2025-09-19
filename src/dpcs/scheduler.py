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
import json
import math
import os

import torch
import torch.nn as nn

from .config import DPCSConfig, CheckpointCfg
from .policies import PrecisionPolicy, CheckpointPolicy
from .signals import GradSignals, CurvatureSignals, EMA, tensor_bytes
from .runtime import (
    amp_preferred_dtype,
    amp_uses_grad_scaler,
    reset_peak_memory_stats,
    max_memory_allocated,
    mem_get_info,
    headroom_frac,
    te_prepare_fp8_modules,
    te_fp8_autocast,
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

    def forward(self, *args, **kwargs):  # type: ignore[override]
        ctx = te_fp8_autocast(self._fp8_active, self._fp8_recipe)
        with ctx:
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
        for st in self._states:
            orig = getattr(st.wrapper, "mod", None)
            if isinstance(orig, nn.Module):
                self._by_orig[orig] = st

    def __getitem__(self, key: nn.Module) -> _LeafState:
        st = self._by_wrapper.get(key)
        if st is None:
            st = self._by_orig.get(key)
        if st is None:
            raise KeyError(key)
        return st

    def __iter__(self):  # type: ignore[override]
        return iter(self._by_wrapper.keys())

    def __len__(self) -> int:  # type: ignore[override]
        return len(self._by_wrapper)

    def __contains__(self, key: object) -> bool:
        return key in self._by_wrapper or key in self._by_orig

    def items(self):  # type: ignore[override]
        return self._by_wrapper.items()

    def values(self):  # type: ignore[override]
        return self._by_wrapper.values()

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

    def __init__(self, **kwargs: Any) -> None:
        ckpt_cfg_in = kwargs.pop("checkpoint_cfg", None)
        self.cfg = DPCSConfig.from_kwargs(**kwargs)
        self._ckpt_cfg = self._normalize_checkpoint_cfg(ckpt_cfg_in)
        self._step = 0

        self._precision_enabled = bool(self.cfg.enable_precision)
        self._manual_precision_mode: Optional[str] = None
        self._headroom_override: Optional[Callable[[], Optional[float]]] = None
        self._compat_states: List[_LeafState] = []
        self._registry: Mapping[nn.Module, _LeafState] = _StateRegistry([])
        self._leaf_names: List[str] = []

        # AMP mode string: "fp32" | "bf16" | "fp16" (with optional fp8 runtime wiring)
        self._amp_preferred_dtype = amp_preferred_dtype(self.cfg.device_type)
        if self.cfg.device_type == "cuda" and torch.cuda.is_available():
            if self._amp_preferred_dtype is torch.bfloat16:
                self._amp_mode = "bf16"
            elif self._amp_preferred_dtype is torch.float16:
                self._amp_mode = "fp16"
            else:
                self._amp_mode = "fp16"
        else:
            self._amp_mode = "fp32"
        self._amp_cooldown = 0

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
        bf16_supported = (
            self.cfg.device_type == "cuda"
            and torch.cuda.is_available()
            and self._amp_preferred_dtype is torch.bfloat16
        )
        self._prec_pol = PrecisionPolicy(self.cfg, bf16_supported=bf16_supported, fp8_supported=False)
        self._ckpt_pol = CheckpointPolicy(self.cfg, self._ckpt_cfg)

        # Overflow tracking
        self._last_scale: Optional[float] = None

        # Logging
        self._log_path: Optional[str] = None
        self._log_every = max(1, int(self.cfg.log_every))

    # ------------------------------ API ------------------------------------

    def wrap(self, model: nn.Module) -> nn.Module:
        model, leaves = _wrap_leaves(model, self._ckpt_cfg)
        self._model = model
        self._leaves = leaves
        self._leaf_names = [w.leaf_name for w in leaves]
        self._compat_states = [_LeafState(self, i, w) for i, w in enumerate(leaves)]
        self._registry = _StateRegistry(self._compat_states)

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
        self._prec_pol.update_fp8_support(self._fp8_supported)

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
        self._log_path = str(path)
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        except Exception:
            pass

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

    def precision_mix(self) -> Dict[str, int]:
        mode = self._effective_amp_mode()
        total = len(self._compat_states) if self._compat_states else len(self._leaves)
        if total <= 0:
            return {}
        return {mode: total}

    # ------------------------- compatibility helpers ----------------------

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

    def force_fp32(self) -> None:
        self.force_precision("fp32")

    def clear_precision_override(self) -> None:
        self._manual_precision_mode = None
        if not self._precision_enabled:
            self._manual_precision_mode = "fp32"
            self._amp_mode = "fp32"

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

        # 2) Detect AMP overflow via GradScaler scale drop
        overflow = False
        if scaler is not None and getattr(scaler, "is_enabled", lambda: False)():
            try:
                curr = float(scaler.get_scale())
                if self._last_scale is not None and curr < self._last_scale:
                    overflow = True
                self._last_scale = curr
            except Exception:
                pass

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

        # 5) Logging (optional)
        if self._log_path and (self._step % self._log_every == 0):
            rec = {
                "step": int(self._step),
                "headroom_frac": float(hr) if hr is not None else None,
                "amp_mode": self._amp_mode,
                "overflow": bool(overflow),
                "ckpt_on": int(sum(1 for w in self._leaves if w.use_ckpt)),
                "num_leaves": int(len(self._leaves)),
                "peak_alloc_bytes": int(peak_b),
                "free_bytes": int(free_b) if free_b is not None else 0,
                "total_bytes": int(total_b) if total_b is not None else 0,
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
                with open(self._log_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
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


__all__ = ["DPCS"]
