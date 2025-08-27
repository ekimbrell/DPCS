"""
DPCS — Dynamic Precision & Checkpointing Scheduler

This version includes:
- Stable Welford variance per step (smoothed with EMA) and curvature proxy.
- Warm‑up auto‑tuner for epsilon_g / kappa (percentile‑based).
- Activation‑aware checkpointing with size gate and requires_grad safety.
- Optional FP8 via Transformer Engine when available.

**Library-side fix implemented:** when `enable_precision=False`, modules start in
FP32 and *do not* enter autocast (CPU bf16 / CUDA fp16). That prevents dtype
mismatches in backward for CPU autocast. Checkpointing can still be used in FP32.
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
    if torch.is_tensor(obj):
        return int(obj.numel()) * int(obj.element_size())
    if isinstance(obj, (list, tuple)):
        return sum(_estimate_bytes(x) for x in obj)
    if isinstance(obj, dict):
        return sum(_estimate_bytes(v) for v in obj.values())
    return 0


def _percentile(xs: list[float], q: float) -> Optional[float]:
    if not xs:
        return None
    try:
        t = torch.tensor(xs, dtype=torch.float64)
        qv = torch.quantile(t, torch.tensor([q], dtype=torch.float64)).item()
        return float(qv)
    except Exception:
        xs_sorted = sorted(xs)
        i = max(0, min(len(xs_sorted) - 1, int(round(q * (len(xs_sorted) - 1)))))
        return float(xs_sorted[i])


# --- Internal state ----------------------------------------------------------

@dataclass
class _ModuleState:
    mode: str = "fp16"        # 'fp32' | 'fp16' | 'fp8'
    cool: int = 0

    # gradient statistics (across steps)
    gvar_ema: Optional[float] = None          # EMA of per-step variance
    grad_l2_ema: Optional[float] = None       # EMA of E[g^2]
    curv_ema: Optional[float] = None          # EMA of |Δ log ||g|| |

    # activation footprint
    last_act_bytes: Optional[int] = None

    # last-step raw stats (for warm-up collection)
    last_var_step: Optional[float] = None
    last_curv_step: Optional[float] = None

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
    epsilon_g: float = 1e-4               # variance threshold (auto-tuned if enabled)
    kappa: float = 0.10                   # curvature threshold (auto-tuned if enabled)
    cooldown_steps: int = 3               # steps to hold FP32 after instability
    ema_beta: float = 0.9                 # EMA smoothing across steps
    signals_freq_steps: int = 1           # compute stats every N steps

    # warm-up auto-tuner
    autotune_precision: bool = True
    autotune_warmup_steps: int = 128
    autotune_gvar_percentile: float = 0.25   # εg ← P25(gvar)
    autotune_curv_percentile: float = 0.75   # κ  ← P75(curv)
    autotune_min_eps: float = 1e-8
    autotune_min_kappa: float = 1e-6

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

        # warm-up collections (global pool across modules)
        self._auto_enabled = bool(self.cfg.autotune_precision) and self.cfg.autotune_warmup_steps > 0
        self._auto_done = False
        self._auto_gvar_samples: list[float] = []
        self._auto_curv_samples: list[float] = []

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
            "eps_g": float(self.cfg.epsilon_g),
            "kappa": float(self.cfg.kappa),
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
        self._swap_model_fp8_modules(model)

        wrap_types: Tuple[Type[nn.Module], ...] = self.cfg.wrap_types
        if self._fp8_ok:
            try:
                wrap_types = tuple(set(list(wrap_types) + [self._te.Linear]))  # type: ignore
            except Exception:
                pass

        for m in model.modules():
            if isinstance(m, wrap_types) and not hasattr(m, "_dpcs_orig_forward"):
                # LIB FIX: initialize FP32 mode when precision is disabled
                initial_mode = "fp32" if not self._prec_on else "fp16"
                self._registry[m] = _ModuleState(mode=initial_mode)
                m._dpcs_orig_forward = m.forward  # type: ignore[attr-defined]

                def make_fwd(mod: nn.Module):
                    def fwd(*args, **kwargs):
                        mode = self._mode_freeze.get(mod, self._registry[mod].mode)

                        def _run(*a):
                            return mod._dpcs_orig_forward(*a, **kwargs)  # type: ignore[attr-defined]

                        want_ckpt = (
                            self._ckpt_on
                            and _args_require_grad(args, kwargs)
                            and self._activation_big_enough(mod)
                        )

                        # Determine if autocast should be used at all
                        enable_amp = self._prec_on and (mode != "fp32")

                        # FP8 path only when precision policy is enabled
                        if mode == "fp8" and self._fp8_ok and self._prec_on:
                            autocast_ctx = self._te.fp8_autocast(
                                enabled=True,
                                fp8_recipe=self._fp8_recipe,
                            )
                            with autocast_ctx:
                                out = checkpoint(_run, *args, use_reentrant=self.cfg.ckpt_use_reentrant) if want_ckpt else mod._dpcs_orig_forward(*args, **kwargs)  # type: ignore[attr-defined]
                            self._registry[mod].last_act_bytes = _estimate_bytes(out)
                            return out

                        # FP16/BF16 autocast branch (only if enabled)
                        if enable_amp:
                            with torch.autocast(self.device_type, dtype=self._autocast_dtype, enabled=True):
                                out = checkpoint(_run, *args, use_reentrant=self.cfg.ckpt_use_reentrant) if want_ckpt else mod._dpcs_orig_forward(*args, **kwargs)  # type: ignore[attr-defined]
                            self._registry[mod].last_act_bytes = _estimate_bytes(out)
                            return out

                        # FP32 path (no autocast)
                        out = checkpoint(_run, *args, use_reentrant=self.cfg.ckpt_use_reentrant) if want_ckpt else mod._dpcs_orig_forward(*args, **kwargs)  # type: ignore[attr-defined]
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

            # --- Welford intra-step variance over module grads ---
            try:
                n = 0
                mean = 0.0
                M2 = 0.0
                for p in mod.parameters(recurse=False):
                    if p.grad is None:
                        continue
                    g = p.grad.detach().to(dtype=torch.float32)
                    k = g.numel()
                    if k == 0:
                        continue
                    m = float(g.mean().item())
                    v = float(g.var(unbiased=False).item())  # population var
                    M2_chunk = v * k
                    delta = m - mean
                    tot = n + k
                    mean = mean + delta * (k / (tot + 1e-12))
                    M2 = M2 + M2_chunk + (delta * delta) * (n * k / (tot + 1e-12))
                    n = tot
                if n > 0:
                    var_step = M2 / max(1, n)
                    st.last_var_step = var_step
                    eg2 = var_step + mean * mean
                    # curvature proxy from E[g^2]
                    if st.grad_l2_ema is not None:
                        curv_now = abs(math.log(eg2 + 1e-12) - math.log(st.grad_l2_ema + 1e-12))
                    else:
                        curv_now = 0.0
                    st.grad_l2_ema = _ema(st.grad_l2_ema, eg2, self.cfg.ema_beta)
                    st.curv_ema = _ema(st.curv_ema, curv_now, self.cfg.ema_beta)
                    st.last_curv_step = curv_now
                    st.gvar_ema = _ema(st.gvar_ema, var_step, self.cfg.ema_beta)

                    if self._auto_enabled and not self._auto_done:
                        self._auto_gvar_samples.append(float(var_step))
                        self._auto_curv_samples.append(float(curv_now))
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
                                    n_el = p.numel()
                                    v_parts.append(v[ofs:ofs+n_el].view_as(p))
                                    ofs += n_el
                                hv = torch.autograd.grad(
                                    grads, params, grad_outputs=v_parts, retain_graph=True, allow_unused=True
                                )
                                hv = [h if h is not None else torch.zeros_like(p) for h, p in zip(hv, params)]
                                hv_flat = torch.cat([h.reshape(-1) for h in hv])
                                hv_trace += float((hv_flat * v).sum().item())
                            hv_trace /= float(samples * (g_flat.numel() + 1e-12))
                            st.curv_ema = _ema(st.curv_ema, abs(hv_trace), self.cfg.ema_beta)
                            if self._auto_enabled and not self._auto_done:
                                self._auto_curv_samples.append(float(abs(hv_trace)))
                except Exception:
                    pass

    # ----- policy application ------------------------------------------------

    def end_step(self, optim: torch.optim.Optimizer, scaler: Optional[torch.amp.GradScaler] = None) -> None:
        # 0) Warm-up auto-tune at boundary
        if self._auto_enabled and not self._auto_done and self._step >= int(self.cfg.autotune_warmup_steps):
            gq = _percentile(self._auto_gvar_samples, float(self.cfg.autotune_gvar_percentile))
            cq = _percentile(self._auto_curv_samples, float(self.cfg.autotune_curv_percentile))
            if gq is not None and math.isfinite(gq):
                self.cfg.epsilon_g = max(float(self.cfg.autotune_min_eps), float(gq))
            if cq is not None and math.isfinite(cq):
                self.cfg.kappa = max(float(self.cfg.autotune_min_kappa), float(cq))
            self._auto_done = True
            self._emit_log({
                "event": "autotune_done",
                "step": int(self._step),
                "epsilon_g": float(self.cfg.epsilon_g),
                "kappa": float(self.cfg.kappa),
                "gvar_samples": len(self._auto_gvar_samples),
                "curv_samples": len(self._auto_curv_samples),
            })
            self._auto_gvar_samples.clear()
            self._auto_curv_samples.clear()

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
            if st.cool > 0:
                st.mode = "fp32"
                if st.just_set_cooldown:
                    st.just_set_cooldown = False
                else:
                    st.cool = max(st.cool - 1, 0)
                continue

            if force_fp32_global or st.pending_overflow or self._has_nonfinite_grad(m):
                st.mode = "fp32"
                st.cool = max(st.cool, self._cool)
                st.just_set_cooldown = True
                st.pending_overflow = False
                continue

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

        self._emit_log({
            "step": int(self._step),
            "headroom": float(self._headroom),
            "ckpt_on": bool(self._ckpt_on),
            "mix": self.precision_mix(),
            "eps_g": float(self.cfg.epsilon_g),
            "kappa": float(self.cfg.kappa),
        })

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
