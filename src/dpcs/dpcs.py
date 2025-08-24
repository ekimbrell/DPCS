# dpcs.py
# Dynamic Precision + Checkpointing Scheduler (DPCS)
# Minimal, practical seed. PyTorch is the only hard dependency.
# Optional: NVIDIA Transformer Engine (TE) for FP8 if installed.

from __future__ import annotations
import contextlib
import functools
import math
import types
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Iterable, Callable

import torch
import torch.nn as nn
from torch.utils.checkpoint import create_selective_checkpoint_contexts, CheckpointPolicy


# -------------------------------
# Optional backends
# -------------------------------
try:
    import transformer_engine.pytorch as te  # type: ignore
    from transformer_engine.common.recipe import DelayedScaling, Format  # type: ignore
    _TE_AVAILABLE = True
except Exception:
    te = None
    DelayedScaling = None
    Format = None
    _TE_AVAILABLE = False


# -------------------------------
# Utilities
# -------------------------------

def _leaf_modules(model: nn.Module) -> Iterable[Tuple[str, nn.Module]]:
    """Yield only leaf modules (no children)."""
    for name, m in model.named_modules():
        if len(list(m.children())) == 0:
            yield name, m


def _flatten_grad_norm_sq(module: nn.Module) -> float:
    """Sum of squared gradient elements across all params of a module (current step)."""
    total = 0.0
    for p in module.parameters(recurse=False):
        if p.grad is not None:
            g = p.grad.detach()
            total += float(g.pow(2).sum().item())
    return total


def _flatten_grad_var(module: nn.Module) -> float:
    """Cheap proxy: variance of gradients across params/elements for this step."""
    # NOTE: This is not per-sample variance; it’s an inexpensive per-parameter-element variance proxy.
    vals = []
    for p in module.parameters(recurse=False):
        if p.grad is not None:
            g = p.grad.detach().float().view(-1)
            if g.numel() > 0:
                vals.append(g.var(unbiased=False).item())
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


# -------------------------------
# Memory monitor
# -------------------------------

@dataclass
class MemoryStats:
    free_bytes: int
    total_bytes: int
    peak_alloc_bytes: int

    @property
    def headroom_bytes(self) -> int:
        return max(self.free_bytes, 0)

    @property
    def headroom_ratio(self) -> float:
        return self.headroom_bytes / max(self.total_bytes, 1)


class MemoryMonitor:
    def __init__(self, device: torch.device | int | None = None):
        self.device = device

    def start_step(self) -> None:
        # Reset peak stats at the beginning of each step.
        try:
            torch.cuda.memory.reset_peak_memory_stats(self.device)
        except Exception:
            # Fallback for older builds
            try:
                torch.cuda.reset_peak_memory_stats(self.device)  # deprecated alias
            except Exception:
                pass

    def end_step(self) -> MemoryStats:
        # Global free / total from cudaMemGetInfo
        try:
            free_b, total_b = torch.cuda.memory.mem_get_info(self.device)
        except Exception:
            # Last-ditch fallback if the memory API is unavailable
            props = torch.cuda.get_device_properties(self.device or torch.cuda.current_device())
            total_b = getattr(props, "total_memory", 0)
            # We can't know "free", so we at least report reserved headroom
            free_b = max(total_b - torch.cuda.memory_reserved(self.device), 0)

        # Peak allocated bytes within the step
        try:
            peak_alloc = torch.cuda.memory.max_memory_allocated(self.device)
        except Exception:
            peak_alloc = torch.cuda.max_memory_allocated(self.device or torch.cuda.current_device())

        return MemoryStats(
            free_bytes=int(free_b),
            total_bytes=int(total_b),
            peak_alloc_bytes=int(peak_alloc),
        )


# -------------------------------
# Precision modes
# -------------------------------

class PrecisionMode:
    FP32 = "fp32"
    FP16 = "fp16"
    FP8  = "fp8"


# -------------------------------
# Per-module state we track
# -------------------------------

@dataclass
class ModuleSignals:
    grad_norm_sq_prev: float = 0.0
    grad_norm_sq_ema: float = 0.0
    grad_var_ema: float = 0.0
    curvature_proxy_ema: float = 0.0
    mode: str = PrecisionMode.FP32
    cooldown_steps: int = 0  # when overflow risk triggers fallback


# -------------------------------
# Precision wrapper (monkey-patch forward)
# -------------------------------

class _ForwardShim:
    """
    Monkey-patches a module's forward to run under the precision mode chosen by the scheduler.
    Keeps the original forward in m._dpcs_orig_forward.
    """

    def __init__(self, module: nn.Module, name: str, registry: Dict[str, ModuleSignals],
                 device_type: str = "cuda",
                 te_recipe=None,
                 allow_fp8: bool = False):
        self.m = module
        self.name = name
        self.registry = registry
        self.device_type = device_type
        self.te_recipe = te_recipe
        self.allow_fp8 = allow_fp8 and _TE_AVAILABLE

        if not hasattr(self.m, "_dpcs_orig_forward"):
            self.m._dpcs_orig_forward = self.m.forward  # type: ignore[attr-defined]
            self.m.forward = types.MethodType(self._wrapped_forward, self.m)  # type: ignore[assignment]

    def _wrapped_forward(self, module, *args, **kwargs):
        state = self.registry[self.name]
        mode = state.mode

        # FP8 region (Transformer Engine) if available and allowed.
        if mode == PrecisionMode.FP8 and self.allow_fp8 and te is not None:
            with te.fp8_autocast(enabled=True, fp8_recipe=self.te_recipe):
                return module._dpcs_orig_forward(*args, **kwargs)  # type: ignore[attr-defined]

        # FP16 region via torch.amp.autocast
        if mode == PrecisionMode.FP16:
            with torch.amp.autocast(self.device_type, dtype=torch.float16):
                return module._dpcs_orig_forward(*args, **kwargs)  # type: ignore[attr-defined]

        # FP32: run without autocast
        return module._dpcs_orig_forward(*args, **kwargs)  # type: ignore[attr-defined]


# -------------------------------
# Checkpoint policy controller
# -------------------------------

class CheckpointPolicyController:
    def __init__(self, headroom_low: float = 0.15, headroom_high: float = 0.35):
        """
        headroom_* are fractions of total device memory. Below low → recompute more.
        Above high → prefer save (less recompute).
        """
        self.headroom_low = headroom_low
        self.headroom_high = headroom_high
        self._last_policy: CheckpointPolicy = CheckpointPolicy.PREFER_SAVE

    def choose_policy(self, mem: MemoryStats) -> CheckpointPolicy:
        r = mem.headroom_ratio
        if r < self.headroom_low:
            self._last_policy = CheckpointPolicy.MUST_RECOMPUTE
        elif r > self.headroom_high:
            self._last_policy = CheckpointPolicy.PREFER_SAVE
        # otherwise keep last policy to avoid flapping
        return self._last_policy

    def context_fn(self, policy: CheckpointPolicy):
        def policy_fn(ctx, op, *args, **kwargs):
            # Use a coarse knob: one policy for all ops this step.
            return policy
        return functools.partial(create_selective_checkpoint_contexts, policy_fn)

    @contextlib.contextmanager
    def contexts(self, mem: MemoryStats):
        policy = self.choose_policy(mem)
        fwd_ctx, bwd_ctx = self.context_fn(policy)()
        with fwd_ctx:
            with contextlib.ExitStack() as stack:
                # Backward context is set by torch when recomputing.
                yield


# -------------------------------
# The DPCS orchestrator
# -------------------------------

@dataclass
class DPCS:
    epsilon_g: float = 1e-3      # gradient variance threshold (lower → allow lower precision)
    kappa: float = 5.0           # curvature threshold (higher → bump back to FP32)
    ema_beta: float = 0.9        # smoothing for signal EMAs
    cooldown_steps: int = 50     # steps to force FP32 after an overflow event
    fp8_backend: Optional[str] = "te"
    signals_freq_steps: int = 10 # compute "heavier" signals every N steps
    device_type: str = "cuda"

    # Runtime
    _registry: Dict[str, ModuleSignals] = field(default_factory=dict, init=False)
    _step: int = field(default=0, init=False)
    _mem: MemoryMonitor = field(default_factory=MemoryMonitor, init=False)
    _ckpt_ctrl: CheckpointPolicyController = field(default_factory=CheckpointPolicyController, init=False)
    _te_recipe = None
    _last_scale: Optional[float] = None
    _overflow_recent: bool = False

    def __post_init__(self):
        if self.fp8_backend == "te" and _TE_AVAILABLE:
            # Reasonable default: Hybrid FP8 (E4M3 fwd, E5M2 bwd) delayed scaling
            self._te_recipe = DelayedScaling(fp8_format=Format.HYBRID, amax_history_len=16, amax_compute_algo="max")

    # ---- Public API ---------------------------------------------------------

    def wrap(self, model: nn.Module, allow_fp8: bool = True) -> nn.Module:
        """
        Register leaf modules and monkey-patch forwards with precision shims.
        If TE is enabled, FP8 regions only help if you’re using TE modules (e.g., te.Linear).
        """
        self._registry.clear()
        for name, m in _leaf_modules(model):
            self._registry[name] = ModuleSignals()
            _ForwardShim(
                m, name, self._registry,
                device_type=self.device_type,
                te_recipe=self._te_recipe,
                allow_fp8=(allow_fp8 and self.fp8_backend == "te")
            )
        return model

    def memory_monitor(self) -> MemoryMonitor:
        return self._mem

    @contextlib.contextmanager
    def checkpoint_context(self):
        # Choose based on last measured memory (updated in end_step),
        # but we still create a valid context; we’ll re-evaluate after forward as needed.
        # Here, we default to PREFER_SAVE until end_step computes true headroom.
        fwd_ctx, _ = create_selective_checkpoint_contexts(lambda *_: CheckpointPolicy.PREFER_SAVE)
        with fwd_ctx[0] if isinstance(fwd_ctx, tuple) else fwd_ctx:
            yield

    # ---- Step lifecycle -----------------------------------------------------

    def start_step(self):
        self._overflow_recent = False
        self._mem.start_step()

    def collect_signals(self, loss: torch.Tensor, model: nn.Module):
        """Should be called after loss.backward(). Updates EMAs and cheap curvature proxy."""
        heavy = (self._step % self.signals_freq_steps == 0)

        for name, m in _leaf_modules(model):
            st = self._registry[name]
            # gradient energy & variance (cheap)
            gn2 = _flatten_grad_norm_sq(m)
            var_now = _flatten_grad_var(m)
            st.grad_norm_sq_ema = self.ema_beta * st.grad_norm_sq_ema + (1 - self.ema_beta) * gn2
            st.grad_var_ema = self.ema_beta * st.grad_var_ema + (1 - self.ema_beta) * var_now

            # curvature proxy: relative change in grad energy
            if st.grad_norm_sq_prev > 0.0:
                rel = abs(gn2 - st.grad_norm_sq_prev) / (st.grad_norm_sq_prev + 1e-12)
            else:
                rel = 0.0
            st.curvature_proxy_ema = self.ema_beta * st.curvature_proxy_ema + (1 - self.ema_beta) * rel
            st.grad_norm_sq_prev = gn2

            # cooldown ticks down
            if st.cooldown_steps > 0:
                st.cooldown_steps -= 1

        # (Optional) Place to add heavier signals (e.g., Hutchinson HVP or BackPACK diag-GGN) when heavy==True

    def end_step(self, optimizer: torch.optim.Optimizer, scaler: Optional[torch.amp.GradScaler] = None):
        """Update precision and checkpoint policy for the *next* step."""
        self._step += 1

        # Overflow detection via GradScaler scale drops (public signal)
        if scaler is not None:
            current_scale = float(scaler.get_scale())
            if self._last_scale is not None and current_scale < self._last_scale:
                self._overflow_recent = True
            self._last_scale = current_scale

        # Memory stats
        mem = self._mem.end_step()

        # Decide checkpoint policy for next step (coarse knob)
        self._ckpt_policy = self._ckpt_ctrl.choose_policy(mem)

        # Per-module precision decisions
        for name, st in self._registry.items():
            # If overflow recently or on cooldown → FP32
            if self._overflow_recent or st.cooldown_steps > 0:
                st.mode = PrecisionMode.FP32
                # Extend cooldown a bit if we actually overflowed
                if self._overflow_recent:
                    st.cooldown_steps = max(st.cooldown_steps, self.cooldown_steps)
                continue

            # High curvature → FP32
            if st.curvature_proxy_ema > self.kappa:
                st.mode = PrecisionMode.FP32
                continue

            # Low gradient variance → allow lower precision
            if st.grad_var_ema < self.epsilon_g:
                # Prefer FP8 if backend is enabled; else FP16
                if self.fp8_backend == "te" and _TE_AVAILABLE:
                    st.mode = PrecisionMode.FP8
                else:
                    st.mode = PrecisionMode.FP16
                continue

            # Default: FP16 for speed, unless signals suggest otherwise
            st.mode = PrecisionMode.FP16

    # ---- Helpers for users --------------------------------------------------

    

    def checkpoint_contexts(self):
        """Context managers for selective checkpointing using the *current* policy."""
        policy = getattr(self, "_ckpt_policy", CheckpointPolicy.PREFER_SAVE)

        def policy_fn(ctx, op, *args, **kwargs):
            # You could branch on `op` or `ctx.is_recompute` here if you want.
            return policy

        return create_selective_checkpoint_contexts(policy_fn)


# -------------------------------
# Example usage (put in your train script)
# -------------------------------
"""
from dpcs import DPCS
import torch, torch.nn as nn

model = nn.Sequential(nn.Linear(4096, 4096), nn.GELU(), nn.Linear(4096, 4096)).cuda()
dpcs = DPCS(epsilon_g=1e-3, kappa=5.0, fp8_backend="te")  # FP8 path auto-enables only if Transformer Engine is installed
model = dpcs.wrap(model, allow_fp8=True)

optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
scaler = torch.amp.GradScaler("cuda")  # unified AMP API

for step in range(10000):
    dpcs.start_step()
    optim.zero_grad(set_to_none=True)

    # (Optional) Use selective activation checkpointing:
    fwd_ctx, bwd_ctx = dpcs.checkpoint_contexts()
    with fwd_ctx:
        x = torch.randn(32, 4096, device="cuda")
        with torch.autocast(device_type="cuda"):  # global safety; per-module DPCS shims will override locally
            y = model(x)
            loss = (y**2).mean()

    scaler.scale(loss).backward()
    dpcs.collect_signals(loss, model)

    scaler.step(optim)
    scaler.update()
    dpcs.end_step(optim, scaler)
"""
