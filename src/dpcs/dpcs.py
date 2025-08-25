# src/dpcs/dpcs.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, Callable
from contextlib import nullcontext
from collections import Counter

import torch
import torch.nn as nn

def summary(self):
    modes = Counter(getattr(st, "mode", "fp16") for st in self._registry.values())
    return {"step": self._step, "modes": dict(modes),
            "ckpt_on": self.is_checkpointing(),
            "headroom": round(self.last_headroom(), 3)}

# Selective activation checkpointing (policy-based)
try:
    from torch.utils.checkpoint import create_selective_checkpoint_contexts, CheckpointPolicy
except Exception:
    create_selective_checkpoint_contexts = None
    class CheckpointPolicy:
        MUST_RECOMPUTE = object()

@dataclass
class _ModuleState:
    name: str
    orig_forward: Callable
    mode: str = "fp16"                 # 'fp32' | 'fp16' | 'fp8'
    grad_var_ema: float = 0.0
    curvature_proxy_ema: float = 0.0
    grad_norm_sq_prev: float = 0.0

def _leaf_modules(model: nn.Module):
    for name, m in model.named_modules():
        if len(list(m.children())) == 0 and any(p.requires_grad for p in m.parameters(recurse=False)):
            yield name, m

@dataclass
class DPCS:
    # policy knobs
    epsilon_g: float = 1e-3            # grad-variance threshold εg
    kappa: float = 5.0                 # curvature threshold κ
    device_type: str = "cuda"
    signals_freq_steps: int = 200
    fp8_backend: Optional[str] = None  # "te" enables Transformer Engine if available

    # adaptive checkpointing hysteresis
    ckpt_low: float = 0.12             # turn ON if free/total < 12%
    ckpt_high: float = 0.20            # turn OFF when free/total > 20%
    ckpt_need: int = 2                 # need N consecutive low-headroom steps

    # runtime state
    _step: int = field(default=0, init=False)
    _registry: Dict[str, _ModuleState] = field(default_factory=dict, init=False)
    _ckpt_on: bool = field(default=False, init=False)
    _ckpt_below: int = field(default=0, init=False)
    _last_headroom: float = field(default=1.0, init=False)
    _ckpt_policy: Any = field(default=None, init=False)
    _te: Any = field(default=None, init=False)      # Transformer Engine handle (optional)
    _allow_fp8: bool = field(default=False, init=False)
    _prev_scale: Optional[float] = field(default=None, init=False)  # for overflow detection

    def __post_init__(self) -> None:
        # Optional FP8 backend discovery (Transformer Engine)
        if self.fp8_backend == "te":
            try:
                import transformer_engine.pytorch as te
                self._te = te
            except Exception:
                self._te = None
        self._ckpt_policy = getattr(CheckpointPolicy, "MUST_RECOMPUTE", CheckpointPolicy.MUST_RECOMPUTE)

    # ------------ public API ------------
    def wrap(self, model: nn.Module, allow_fp8: bool = False) -> nn.Module:
        """Install per-leaf forward shims so DPCS can enforce per-layer precision at runtime."""
        self._allow_fp8 = bool(allow_fp8 and (self._te is not None))
        for name, m in _leaf_modules(model):
            if hasattr(m, "_dpcs_state"):
                continue
            st = _ModuleState(name=name, orig_forward=m.forward)
            self._registry[name] = st
            m._dpcs_state = st  # attach
            m.forward = self._make_precision_shim(m, st)  # type: ignore[assignment]
        return model

    def start_step(self) -> None:
        self._step += 1

    def collect_signals(self, loss: torch.Tensor, model: nn.Module) -> None:
        """Update per-module signals. Heavy stats can be computed sparsely."""
        heavy = (self._step % self.signals_freq_steps == 0)
        for name, m in _leaf_modules(model):
            st = self._registry[name]
            # curvature proxy ~ change in grad L2^2
            gn2 = 0.0
            for p in m.parameters(recurse=False):
                if p.grad is not None:
                    gn2 += float(p.grad.detach().pow(2).sum().item())
            rel = abs(gn2 - st.grad_norm_sq_prev) / (st.grad_norm_sq_prev + 1e-12) if st.grad_norm_sq_prev > 0 else 0.0
            st.curvature_proxy_ema = 0.9 * st.curvature_proxy_ema + 0.1 * rel
            st.grad_norm_sq_prev = gn2

            if heavy:
                grads = []
                for p in m.parameters(recurse=False):
                    if p.grad is not None:
                        grads.append(p.grad.detach().float().reshape(-1))
                if grads:
                    g = torch.cat(grads)
                    var_now = float(g.var(unbiased=False).item())
                    st.grad_var_ema = 0.9 * st.grad_var_ema + 0.1 * var_now

    def end_step(self, optimizer: torch.optim.Optimizer, scaler: Optional[torch.amp.GradScaler] = None) -> None:
        """Call once per iteration after optimizer/scaler.step()."""
        # 1) update adaptive checkpointing gate from memory headroom
        self._update_ckpt_gate()

        # 2) detect overflow via GradScaler scale drop
        overflow = False
        if scaler is not None:
            try:
                cur = float(scaler.get_scale())
                if self._prev_scale is not None and cur < self._prev_scale:
                    overflow = True
                self._prev_scale = cur
            except Exception:
                pass

        # 3) precision mode for *next* step
        for st in self._registry.values():
            if overflow:
                st.mode = "fp32"
                continue
            lower_ok = (st.grad_var_ema < self.epsilon_g) or (st.curvature_proxy_ema > self.kappa)
            if self._allow_fp8 and lower_ok:
                st.mode = "fp8"
            elif lower_ok:
                st.mode = "fp16"
            # else keep previous (sticky)

    def checkpoint_contexts_if_needed(self) -> Tuple[object, object]:
        """Return (fwd_ctx, bwd_ctx). Null contexts if gate is off."""
        if (not self._ckpt_on) or (create_selective_checkpoint_contexts is None) or (self._ckpt_policy is None):
            return nullcontext(), nullcontext()
        def _policy_fn(_ctx, _op, *a, **k):
            return self._ckpt_policy
        return create_selective_checkpoint_contexts(_policy_fn)

    def is_checkpointing(self) -> bool:
        return bool(self._ckpt_on)

    def last_headroom(self) -> float:
        return float(self._last_headroom)

    def summary(self) -> Dict[str, Any]:
        modes = Counter(getattr(st, "mode", "fp16") for st in self._registry.values())
        return {"step": self._step, "modes": dict(modes), "ckpt_on": self.is_checkpointing(), "headroom": round(self.last_headroom(), 3)}

    # ------------ internals ------------
    def _make_precision_shim(self, module: nn.Module, st: _ModuleState):
        te = self._te
        def _fwd(*args, **kwargs):
            m = st.mode
            if m == "fp32":
                # force full precision locally (disable autocast)
                with torch.autocast(self.device_type, enabled=False):
                    return st.orig_forward(*args, **kwargs)
            elif m == "fp8" and (te is not None):
                # optional FP8 region (Transformer Engine)
                with te.fp8_autocast(enabled=True):
                    return st.orig_forward(*args, **kwargs)
            else:
                # fp16 path relies on outer global autocast (AMP)
                return st.orig_forward(*args, **kwargs)
        return _fwd

    def _read_headroom(self) -> float:
        if self.device_type == "cuda" and torch.cuda.is_available():
            try:
                free_b, total_b = torch.cuda.memory.mem_get_info()  # (free, total) bytes
                return (float(free_b) / float(total_b)) if total_b else 1.0
            except Exception:
                pass
        return getattr(self, "_last_headroom", 1.0)

    def _update_ckpt_gate(self) -> None:
        r = self._read_headroom()
        self._last_headroom = r
        if r < self.ckpt_low:
            self._ckpt_below += 1
        elif r > self.ckpt_high:
            self._ckpt_below = 0
            self._ckpt_on = False
        if (not self._ckpt_on) and (self._ckpt_below >= self.ckpt_need):
            self._ckpt_on = True
