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

from typing import Any, List, Optional, Tuple
import json
import math
import os
import time

import torch
import torch.nn as nn

from .config import DPCSConfig
from .policies import PrecisionPolicy, CheckpointPolicy
from .signals import GradSignals, CurvatureSignals, EMA, tensor_bytes
from .runtime import (
    amp_preferred_dtype,
    amp_enabled,
    amp_uses_grad_scaler,
    reset_peak_memory_stats,
    max_memory_allocated,
    mem_get_info,
    headroom_frac,
    checkpoint_call,
)

# ------------------------ checkpointable leaf wrapper -----------------------

class _CkptWrap(nn.Module):
    """Wrap a leaf to optionally run under activation checkpointing.

    Also records lightweight telemetry per forward (activation bytes, latency
    EMA). Uses non-reentrant checkpointing with no RNG state stash.
    """
    def __init__(self, mod: nn.Module) -> None:
        super().__init__()
        self.mod = mod
        self.use_ckpt: bool = False
        self._act_bytes: int = 0
        self._t_ema = EMA(beta=0.9)
        if torch.cuda.is_available():
            self._ev_s = torch.cuda.Event(enable_timing=True)
            self._ev_e = torch.cuda.Event(enable_timing=True)
        else:
            self._ev_s = None  # type: ignore
            self._ev_e = None  # type: ignore

    def forward(self, *args, **kwargs):  # type: ignore[override]
        # timing start
        if self._ev_s is not None:
            self._ev_s.record()
        t0 = time.perf_counter()
        if self.use_ckpt:
            out = checkpoint_call(self.mod, *args, **kwargs)
        else:
            out = self.mod(*args, **kwargs)
        # activation bytes and elapsed
        self._act_bytes = tensor_bytes(out)
        if self._ev_e is not None and self._ev_s is not None:
            self._ev_e.record(); self._ev_e.synchronize()
            ms = float(self._ev_s.elapsed_time(self._ev_e))
        else:
            ms = (time.perf_counter() - t0) * 1e3
        self._t_ema.update(ms)
        return out


# --------------------------- leaf discovery / wrap -------------------------

_LEAF_TYPES = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)

def _is_leaf(m: nn.Module) -> bool:
    return (len(list(m.children())) == 0) or isinstance(m, _LEAF_TYPES)


def _wrap_leaves(model: nn.Module) -> Tuple[nn.Module, List[_CkptWrap]]:
    wrappers: List[_CkptWrap] = []
    def _wrap(module: nn.Module) -> nn.Module:
        if _is_leaf(module):
            w = _CkptWrap(module)
            wrappers.append(w)
            return w
        for name, child in list(module.named_children()):
            setattr(module, name, _wrap(child))
        return module
    model = _wrap(model)
    return model, wrappers


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

    def __init__(self, **kwargs: Any) -> None:
        self.cfg = DPCSConfig.from_kwargs(**kwargs)
        self._step = 0

        # AMP mode string: "fp32" | "bf16" | "fp16"
        if self.cfg.device_type == "cuda" and torch.cuda.is_available():
            self._amp_mode = "bf16" if (amp_preferred_dtype("cuda") is torch.bfloat16) else "fp16"
        else:
            self._amp_mode = "fp32"
        self._amp_cooldown = 0

        # Leaves and signals
        self._model: Optional[nn.Module] = None
        self._leaves: List[_CkptWrap] = []
        self._grads: Optional[GradSignals] = None
        self._curv: Optional[CurvatureSignals] = None

        # Policies
        self._prec_pol = PrecisionPolicy(self.cfg, bf16_supported=(amp_preferred_dtype("cuda") is torch.bfloat16))
        self._ckpt_pol = CheckpointPolicy(self.cfg)

        # Overflow tracking
        self._last_scale: Optional[float] = None

        # Logging
        self._log_path: Optional[str] = None
        self._log_every = max(1, int(self.cfg.log_every))

    # ------------------------------ API ------------------------------------

    def wrap(self, model: nn.Module) -> nn.Module:
        model, leaves = _wrap_leaves(model)
        self._model = model
        self._leaves = leaves

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
        if self._amp_mode == "bf16" and (amp_preferred_dtype("cuda") is torch.bfloat16):
            return ("cuda", torch.bfloat16, True)
        if self._amp_mode == "fp16":
            return ("cuda", torch.float16, True)
        return ("cuda", torch.float32, False)

    def amp_uses_grad_scaler(self) -> bool:
        dev, dtype, enabled = self.get_amp_config()
        return enabled and amp_uses_grad_scaler(dtype, dev)

    def start_step(self) -> None:
        # Step-scoped peak VRAM reset
        reset_peak_memory_stats()
        # Decay blacklist counters for ckpt policy
        self._ckpt_pol.tick()

    def collect_signals(self, loss: Optional[torch.Tensor], model: Optional[nn.Module] = None) -> None:
        # Optional curvature probe on schedule (requires loss with graph)
        if loss is not None and self._curv is not None:
            try:
                self._curv.maybe_probe(loss)
            except RuntimeError:
                # Allow training to continue if the current step cannot build HVP
                pass

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
        hr = headroom_frac()
        hr = self._sync_min_headroom(hr)
        free_b, total_b = 0, 1
        info = mem_get_info()
        
        if info is not None:
            free_b, total_b = info

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
        new_mode = self._prec_pol.decide(hr, gvar, curv_avg, overflow, self._amp_mode)
        self._amp_mode = new_mode

        # 4) Checkpoint policy
        if self._leaves:
            act = [w._act_bytes for w in self._leaves]
            fwd = [w._t_ema.value if w._t_ema.initialized else 0.0 for w in self._leaves]
            enable_ids = self._ckpt_pol.plan(act, fwd, hr)
            for i, w in enumerate(self._leaves):
                w.use_ckpt = (i in enable_ids)

        # 5) Logging (optional)
        if self._log_path and (self._step % self._log_every == 0):
            rec = {
                "step": int(self._step),
                "headroom_frac": float(hr) if hr is not None else None,
                "amp_mode": self._amp_mode,
                "overflow": bool(overflow),
                "ckpt_on": int(sum(1 for w in self._leaves if w.use_ckpt)),
                "num_leaves": int(len(self._leaves)),
                "peak_alloc_bytes": int(max_memory_allocated()),
                "free_bytes": int(free_b),
                "total_bytes": int(total_b),
                "grad_var_avg": float(gvar) if gvar is not None else None,
                "curv_avg": float(curv_avg) if curv_avg is not None else None,
            }
            try:
                with open(self._log_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
            except Exception:
                pass

        self._step += 1


__all__ = ["DPCS"]
