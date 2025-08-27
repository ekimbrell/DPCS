"""
DPCS — Dynamic Precision & Checkpointing Scheduler (advanced)

What this file provides (production-ready baseline):
- Per-module precision policy (FP32/FP16/FP8*), with overflow cooldown
- Real gradient variance via EMA of first/second moments
- Curvature proxy via smoothed |Δ log ∥g∥|, optional Hutchinson HVP probe
- Activation checkpointing with:
  • headroom-driven hysteresis (ON/OFF votes)
  • safety gating (only when some input requires_grad)
  • activation-aware filter (only checkpoint modules whose last outputs were big)
- Optional FP8 path via NVIDIA Transformer Engine (TE)

(*) FP8 requires TE and FP8-capable hardware; otherwise falls back to FP16.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, Type, Any

import math
import warnings

import torch
import torch.nn as nn
from contextlib import contextmanager, nullcontext
from torch.utils.checkpoint import checkpoint
from collections import Counter

# --- Optional FP8 backend ----------------------------------------------------
try:
    import transformer_engine.pytorch as te
    from transformer_engine.pytorch import recipe as te_recipe
    _TE_AVAILABLE = True
except Exception:
    te = None  # type: ignore
    te_recipe = None  # type: ignore
    _TE_AVAILABLE = False


# --- Utilities ---------------------------------------------------------------

def _has_cuda() -> bool:
    return torch.cuda.is_available()


def _ema(old: Optional[float], new: float, beta: float) -> float:
    if old is None:
        return float(new)
    return float(beta * float(old) + (1.0 - beta) * float(new))


def _args_require_grad(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> bool:
    for t in args:
        if torch.is_tensor(t) and t.requires_grad:
            return True
    for v in kwargs.values():
        if torch.is_tensor(v) and v.requires_grad:
            return True
    return False


def _estimate_bytes(obj: Any) -> int:
    """Best-effort estimate of bytes held by a (possibly nested) tensor output."""
    if torch.is_tensor(obj):
        return int(obj.numel()) * int(obj.element_size())
    if isinstance(obj, (list, tuple)):
        return sum(_estimate_bytes(x) for x in obj)
    if isinstance(obj, dict):
        return sum(_estimate_bytes(v) for v in obj.values())
    return 0


# --- Internal state ----------------------------------------------------------

@dataclass
class _ModuleState:
    mode: str = "fp16"        # 'fp32' | 'fp16' | 'fp8'
    cool: int = 0

    # gradient statistics (EMAs over steps)
    grad_mean_ema: Optional[float] = None     # E[g]
    grad_sqmean_ema: Optional[float] = None   # E[g^2]
    gvar_ema: Optional[float] = None          # Var[g] ≈ E[g^2] - E[g]^2
    grad_l2_ema: Optional[float] = None       # ||g||^2 / N (per-element)

    # curvature proxy
    curv_ema: Optional[float] = None          # EMA of |Δ log ||g|| |

    # activation footprint (from previous forward)
    last_act_bytes: Optional[int] = None

    # overflow / control flags
    pending_overflow: bool = False
    just_set_cooldown: bool = False


# --- Public config -----------------------------------------------------------

@dataclass
class DPCSConfig:
    # device / wrapping
    device_type: str = "cuda"
    allow_fp8: bool = False
    fp8_backend: str = "te"              # only 'te' supported
    swap_modules_for_fp8: bool = True     # swap nn.Linear → te.Linear when possible

    # precision scheduler knobs
    enable_precision: bool = True
    epsilon_g: float = 1e-4               # variance threshold (tune per model)
    kappa: float = 0.10                   # curvature threshold on |Δ log ||g|| |
    cooldown_steps: int = 3               # steps to hold FP32 after instability
    ema_beta: float = 0.9                 # EMA smoothing for stats
    signals_freq_steps: int = 1           # compute stats every N steps

    # checkpointing hysteresis (based on headroom)
    ckpt_low: float = 0.05                # turn ON ckpt if headroom <= low
    ckpt_high: float = 0.20               # turn OFF ckpt if headroom >= high
    ckpt_need: int = 2                    # consecutive votes required

    # activation-aware checkpointing
    min_activation_bytes_to_ckpt: int = 8 << 20   # 8 MB default
    ckpt_use_reentrant: bool = False              # recommended in recent PyTorch

    # curvature (optional) Hutchinson HVP probe
    curv_hvp_freq: int = 0                # 0 disables; else every N steps try HVP
    curv_hvp_samples: int = 1             # Hutchinson probe samples when enabled

    # which module types we wrap by default (include transformer blocks)
    wrap_types: Tuple[Type[nn.Module], ...] = (nn.Linear, nn.TransformerEncoderLayer)


# --- Main class --------------------------------------------------------------

class DPCS:
    def __init__(self, **kwargs):
        self.cfg = DPCSConfig(**kwargs)
        self.device_type = self.cfg.device_type
        self._prec_on = bool(self.cfg.enable_precision)

        # runtime state
        self._step = 0
        self._last_scale: Optional[float] = None
        self._ckpt_on = False
        self._ckpt_votes = 0
        self._freeze_active: bool = False
        self._mode_freeze: Dict[nn.Module, str] = {}
        self._registry: Dict[nn.Module, _ModuleState] = {}
        self._log_cb: Optional[Callable[[dict], None]] = None
        self._headroom: float = self._snapshot_headroom()
        self._cool: int = int(self.cfg.cooldown_steps)

        # device dtypes
        self._autocast_dtype = (
            torch.float16 if self.device_type == "cuda" else torch.bfloat16
        )

        # FP8 setup (Transformer Engine)
        self._fp8_ok = False
        self._te = None
        self._fp8_recipe = None
        if self.cfg.allow_fp8 and self.cfg.fp8_backend == "te":
            if _TE_AVAILABLE:
                self._fp8_ok = True
                self._te = te
                try:
                    self._fp8_recipe = te_recipe.DelayedScaling()
                except Exception:
                    self._fp8_recipe = None
            else:
                warnings.warn(
                    "allow_fp8=True but Transformer Engine is not available; disabling FP8.",
                    RuntimeWarning,
                )

    # ----- low-level helpers -------------------------------------------------

    def _snapshot_headroom(self) -> float:
        if self.device_type != "cuda" or not _has_cuda():
            return 1.0
        try:
            free, total = torch.cuda.mem_get_info()
            return max(0.0, min(1.0, float(free) / float(total)))
        except Exception:
            return 1.0

    def _has_nonfinite_grad(self, mod: nn.Module) -> bool:
        for p in mod.parameters(recurse=False):
            if p.grad is not None and not torch.isfinite(p.grad).all():
                return True
        return False

    def _emit_log(self, payload: dict) -> None:
        cb = getattr(self, "_log_cb", None)
        if cb is not None:
            try:
                cb(payload)
            except Exception:
                pass

    def on_log(self, fn: Callable[[dict], None]) -> None:
        self._log_cb = fn

    def precision_mix(self) -> Dict[str, int]:
        return dict(Counter(st.mode for st in self._registry.values())) if self._registry else {}

    def log_dict(self) -> dict:
        try:
            alloc = torch.cuda.memory.get_allocator_backend()
        except Exception:
            alloc = "unknown"
        return {
            "step": self._step,
            "ckpt_on": self._ckpt_on,
            "precision_mix": self.precision_mix(),
            "device": str(self.device_type),
            "allocator_backend": alloc,
            "headroom": float(getattr(self, "_headroom", 1.0)),
        }

    def _activation_big_enough(self, mod: nn.Module) -> bool:
        st = self._registry.get(mod)
        if st is None:
            return False
        th = int(self.cfg.min_activation_bytes_to_ckpt)
        return (st.last_act_bytes is not None) and (st.last_act_bytes >= th)

    # ----- FP8 module swap (optional) ---------------------------------------

    def _swap_model_fp8_modules(self, root: nn.Module) -> None:
        if not (self._fp8_ok and self.cfg.swap_modules_for_fp8):
            return

        def _recurse(parent: nn.Module):
            for name, child in list(parent.named_children()):
                _recurse(child)
                if isinstance(child, nn.Linear):
                    te_lin = self._te.Linear(
                        child.in_features,
                        child.out_features,
                        bias=(child.bias is not None),
                        params_dtype=child.weight.dtype,
                        device=child.weight.device,
                    )
                    with torch.no_grad():
                        te_lin.weight.copy_(child.weight)
                        if child.bias is not None:
                            te_lin.bias.copy_(child.bias)
                    setattr(parent, name, te_lin)
        _recurse(root)

    # ----- wrapping ----------------------------------------------------------

    def wrap(self, model: nn.Module) -> nn.Module:
        # Optionally swap modules to FP8-capable TE ops
        self._swap_model_fp8_modules(model)

        # Decide which types to wrap
        wrap_types: Tuple[Type[nn.Module], ...] = self.cfg.wrap_types
        if self._fp8_ok:
            try:
                wrap_types = tuple(set(list(wrap_types) + [self._te.Linear]))  # type: ignore
            except Exception:
                pass

        for m in model.modules():
            if isinstance(m, wrap_types) and not hasattr(m, "_dpcs_orig_forward"):
                self._registry[m] = _ModuleState(mode="fp16")
                m._dpcs_orig_forward = m.forward  # type: ignore[attr-defined]

                def make_fwd(mod: nn.Module):
                    def fwd(*args, **kwargs):
                        mode = self._mode_freeze.get(mod, self._registry[mod].mode)

                        def _run(*a):
                            return mod._dpcs_orig_forward(*a, **kwargs)  # type: ignore[attr-defined]

                        # Should we checkpoint this module on this call?
                        want_ckpt = (
                            self._ckpt_on
                            and _args_require_grad(args, kwargs)
                            and self._activation_big_enough(mod)
                        )

                        # FP32 branch
                        if mode == "fp32":
                            if want_ckpt:
                                out = checkpoint(_run, *args, use_reentrant=self.cfg.ckpt_use_reentrant)
                            else:
                                out = mod._dpcs_orig_forward(*args, **kwargs)  # type: ignore[attr-defined]
                            # record activation size for future decisions
                            self._registry[mod].last_act_bytes = _estimate_bytes(out)
                            return out

                        # FP8 branch (TE)
                        if mode == "fp8" and self._fp8_ok:
                            autocast_ctx = self._te.fp8_autocast(
                                enabled=True,
                                fp8_recipe=self._fp8_recipe,
                            )
                            with autocast_ctx:
                                if want_ckpt:
                                    out = checkpoint(_run, *args, use_reentrant=self.cfg.ckpt_use_reentrant)
                                else:
                                    out = mod._dpcs_orig_forward(*args, **kwargs)  # type: ignore[attr-defined]
                            self._registry[mod].last_act_bytes = _estimate_bytes(out)
                            return out

                        # FP16/BF16 branch
                        with torch.autocast(self.device_type, dtype=self._autocast_dtype, enabled=True):
                            if want_ckpt:
                                out = checkpoint(_run, *args, use_reentrant=self.cfg.ckpt_use_reentrant)
                            else:
                                out = mod._dpcs_orig_forward(*args, **kwargs)  # type: ignore[attr-defined]
                        self._registry[mod].last_act_bytes = _estimate_bytes(out)
                        return out

                    return fwd

                m.forward = make_fwd(m)  # type: ignore[method-assign]
        return model

    # ----- step lifecycle ----------------------------------------------------

    def start_step(self) -> None:
        self._step += 1
        self._headroom = self._snapshot_headroom()

    def checkpoint_contexts_if_needed(self):
        if self._ckpt_on:
            return self._checkpoint_region(), nullcontext()
        return nullcontext(), nullcontext()

    @contextmanager
    def _checkpoint_region(self):
        self._freeze_active = True
        try:
            yield
        finally:
            self._freeze_active = False

    # ----- signal collection -------------------------------------------------

    def collect_signals(self, loss: torch.Tensor, model: torch.nn.Module) -> None:
        if not self._prec_on:
            return

        freq = max(1, int(self.cfg.signals_freq_steps))
        heavy = (self._step % freq == 0)

        for mod, st in self._registry.items():
            # overflow sniff
            try:
                for p in mod.parameters(recurse=False):
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        st.pending_overflow = True
                        break
            except Exception:
                st.pending_overflow = True

            # gradient stats (heavy path optional)
            try:
                g_sum = 0.0
                g_sqsum = 0.0
                count = 0
                for p in mod.parameters(recurse=False):
                    if p.grad is None:
                        continue
                    g = p.grad.detach().to(dtype=torch.float32)
                    g_sum += float(g.sum().item())
                    g_sqsum += float(g.pow(2).sum().item())
                    count += g.numel()
                if count > 0:
                    g_mean = g_sum / count
                    g_sqmean = g_sqsum / count
                    st.grad_mean_ema = _ema(st.grad_mean_ema, g_mean, self.cfg.ema_beta)
                    st.grad_sqmean_ema = _ema(st.grad_sqmean_ema, g_sqmean, self.cfg.ema_beta)
                    var_est = max(0.0, g_sqmean - g_mean * g_mean)
                    st.gvar_ema = _ema(st.gvar_ema, var_est, self.cfg.ema_beta)

                    cur_l2 = g_sqmean
                    if st.grad_l2_ema is not None:
                        dlog = abs(math.log(cur_l2 + 1e-12) - math.log(st.grad_l2_ema + 1e-12))
                        st.curv_ema = _ema(st.curv_ema, dlog, self.cfg.ema_beta)
                    st.grad_l2_ema = _ema(st.grad_l2_ema, cur_l2, self.cfg.ema_beta)
            except Exception:
                pass

            # Optional Hutchinson HVP probe (conservative)
            if heavy and self.cfg.curv_hvp_freq > 0 and (self._step % self.cfg.curv_hvp_freq == 0):
                try:
                    if loss.requires_grad and torch.is_grad_enabled():
                        hv_trace = 0.0
                        samples = max(1, int(self.cfg.curv_hvp_samples))
                        params = [p for p in mod.parameters(recurse=False) if p.requires_grad]
                        if params:
                            grads = [p.grad.detach() if p.grad is not None else torch.zeros_like(p) for p in params]
                            g_flat = torch.cat([g.reshape(-1) for g in grads])
                            for _ in range(samples):
                                v = torch.randint_like(g_flat, low=0, high=2).float() * 2 - 1
                                v_parts = []
                                ofs = 0
                                for p in params:
                                    n = p.numel()
                                    v_parts.append(v[ofs:ofs+n].view_as(p))
                                    ofs += n
                                hv = torch.autograd.grad(
                                    grads, params, grad_outputs=v_parts, retain_graph=True, allow_unused=True
                                )
                                hv = [h if h is not None else torch.zeros_like(p) for h, p in zip(hv, params)]
                                hv_flat = torch.cat([h.reshape(-1) for h in hv])
                                hv_trace += float((hv_flat * v).sum().item())
                            hv_trace /= float(samples * (g_flat.numel() + 1e-12))
                            st.curv_ema = _ema(st.curv_ema, abs(hv_trace), self.cfg.ema_beta)
                except Exception:
                    pass

    # ----- policy application ------------------------------------------------

    def end_step(self, optim: torch.optim.Optimizer, scaler: Optional[torch.amp.GradScaler] = None) -> None:
        # 1) Checkpointing hysteresis votes
        low, high, need = self.cfg.ckpt_low, self.cfg.ckpt_high, self.cfg.ckpt_need
        vote_on = (self._headroom <= low)
        vote_off = (self._headroom >= high)
        if vote_on:
            self._ckpt_votes = max(0, self._ckpt_votes) + 1
        elif vote_off:
            self._ckpt_votes = min(0, self._ckpt_votes) - 1
        if self._ckpt_votes >= need:
            self._ckpt_on = True
            self._ckpt_votes = 0
        if self._ckpt_votes <= -need:
            self._ckpt_on = False
            self._ckpt_votes = 0

        # 2) Global overflow via GradScaler scale drop
        force_fp32_global = False
        if scaler is not None and hasattr(scaler, "get_scale"):
            try:
                cur_scale = float(scaler.get_scale())
                if self._last_scale is not None and cur_scale < float(self._last_scale):
                    force_fp32_global = True
                self._last_scale = cur_scale
            except Exception:
                pass

        # 3) Per-module next-mode decision
        for m, st in self._registry.items():
            # cooldown wins
            if st.cool > 0:
                st.mode = "fp32"
                if st.just_set_cooldown:
                    st.just_set_cooldown = False
                else:
                    st.cool = max(st.cool - 1, 0)
                continue

            # enforce FP32 on overflow
            if force_fp32_global or st.pending_overflow or self._has_nonfinite_grad(m):
                st.mode = "fp32"
                st.cool = max(st.cool, self._cool)
                st.just_set_cooldown = True
                st.pending_overflow = False
                continue

            # variance + curvature policy
            gvar = st.gvar_ema if st.gvar_ema is not None else float("inf")
            curv = st.curv_ema if st.curv_ema is not None else 0.0
            drop = (gvar < self.cfg.epsilon_g) or (curv > self.cfg.kappa)
            if drop:
                if self._fp8_ok and self.cfg.allow_fp8:
                    st.mode = "fp8"
                else:
                    st.mode = "fp16"
            else:
                st.mode = "fp32" if (self._headroom > 0.5) else "fp16"

        # 4) log
        self._emit_log({
            "step": int(self._step),
            "headroom": float(self._headroom),
            "ckpt_on": bool(self._ckpt_on),
            "mix": self.precision_mix(),
        })

        # 5) cleanup
        self._freeze_active = False
        self._mode_freeze.clear()

    # ----- ergonomics --------------------------------------------------------

    def is_checkpointing(self) -> bool:
        return self._ckpt_on

    def modes_summary(self) -> Dict[str, int]:
        return self.precision_mix()


# --- helpers -----------------------------------------------------------------

def _count_modes(reg: Dict[nn.Module, _ModuleState]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for st in reg.values():
        out[st.mode] = out.get(st.mode, 0) + 1
    return out
