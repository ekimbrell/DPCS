from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, Any, Iterable, Type
from collections import Counter
from contextlib import contextmanager, nullcontext
import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

try:
    import transformer_engine.pytorch as te  # Optional FP8
    _TE_AVAILABLE = True
except Exception:
    _TE_AVAILABLE = False

# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------

def _has_cuda() -> bool:
    return torch.cuda.is_available()


def _tensor_bytes(obj: Any) -> int:
    """Best-effort count of tensor payload bytes for nested outputs."""
    if torch.is_tensor(obj):
        return obj.numel() * obj.element_size()
    if isinstance(obj, (list, tuple)):
        return sum(_tensor_bytes(x) for x in obj)
    if isinstance(obj, dict):
        return sum(_tensor_bytes(v) for v in obj.values())
    return 0


def _ema(old: Optional[float], new: float, beta: float) -> float:
    return float(new if old is None else beta * float(old) + (1.0 - beta) * float(new))


# -----------------------------------------------------------------------------
# State
# -----------------------------------------------------------------------------

@dataclass
class _ModuleState:
    # precision policy
    mode: str = "fp16"            # 'fp32' | 'fp16' | 'fp8'
    cool: int = 0                 # cooldown steps after overflow

    # running stats for precision/autotune
    grad_l2_ema: Optional[float] = None
    gvar_ema: Optional[float] = None
    curv_ema: Optional[float] = None

    # activation bookkeeping (for ckpt policy)
    last_act_bytes: int = 0

    # overflow/guard rails
    pending_overflow: bool = False
    just_set_cooldown: bool = False

    # harmful checkpoint detection
    harmful_count: int = 0
    ckpt_blacklisted: bool = False


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

@dataclass
class DPCSConfig:
    # device / wrapping
    device_type: str = "cuda"

    # FP8
    allow_fp8: bool = False
    fp8_backend: str = "te"

    # precision scheduler knobs
    enable_precision: bool = True
    epsilon_g: float = 1e-3
    kappa: float = 5.0
    cooldown_steps: int = 3
    ema_beta: float = 0.9

    # signals compute cadence (1 = every step)
    signals_freq_steps: int = 1

    # checkpointing policy
    min_activation_bytes_to_ckpt: int = 16 << 20   # 16 MiB gate
    ckpt_low: float = 0.05                         # headroom hysteresis (not used here directly)
    ckpt_high: float = 0.20
    ckpt_need: int = 2
    ckpt_harmful_delta_bytes: int = 8 << 20        # if local peak rises > 8 MiB while ckpt'ing this block
    ckpt_harmful_patience: int = 3                  # consecutive steps before blacklisting

    # which module types we precision-wrap vs ckpt-wrap
    wrap_types: Tuple[Type[nn.Module], ...] = (nn.Linear, nn.TransformerEncoderLayer)
    ckpt_wrap_types: Tuple[Type[nn.Module], ...] = (nn.Sequential, nn.TransformerEncoderLayer)


# -----------------------------------------------------------------------------
# Main scheduler
# -----------------------------------------------------------------------------

class DPCS:
    def __init__(self, **kwargs):
        self.cfg = DPCSConfig(**kwargs)
        self.device_type = self.cfg.device_type

        # state
        self._step = 0
        self._ckpt_on = False
        self._ckpt_votes = 0
        self._registry: Dict[nn.Module, _ModuleState] = {}
        self._mode_freeze: Dict[nn.Module, str] = {}
        self._freeze_active: bool = False
        self._log_cb: Optional[Callable[[dict], None]] = None

        # AMP scale tracking (overflow hint)
        self._last_scale: Optional[float] = None

        # FP8 gate
        self._fp8_ok = False
        self._te = None
        if self.cfg.allow_fp8 and self.cfg.fp8_backend == "te" and _TE_AVAILABLE:
            self._fp8_ok = True
            self._te = te

        # precision toggle determines autocast use; stats are always collected
        self._prec_on = bool(self.cfg.enable_precision)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def wrap(self, model: nn.Module) -> nn.Module:
        for m in model.modules():
            if isinstance(m, self.cfg.wrap_types) and not hasattr(m, "_dpcs_orig_forward"):
                self._registry[m] = _ModuleState(mode="fp16")
                m._dpcs_orig_forward = m.forward  # type: ignore[attr-defined]

                def make_fwd(mod: nn.Module):
                    st = self._registry[mod]

                    def _needs_rng_state(mod_: nn.Module) -> bool:
                        # fast scan for stochastic ops (dropout); if any active, preserve RNG
                        for sub in mod_.modules():
                            if isinstance(sub, (nn.Dropout, nn.AlphaDropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)) and sub.p > 0 and sub.training:
                                return True
                            # common patterns exposing 'dropout' float on modules (e.g., Transformer)
                            if hasattr(sub, "dropout") and isinstance(getattr(sub, "dropout"), float) and getattr(sub, "dropout") > 0 and sub.training:
                                return True
                        return False

                    def _local_autocast_enabled() -> bool:
                        # precision scheduling off => always run FP32 (no autocast)
                        if not self._prec_on:
                            return False
                        mode = self._mode_freeze.get(mod, st.mode)
                        return mode in ("fp16", "fp8")

                    def _local_dtype():
                        # prefer bf16 when available for better stability; else fp16
                        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                            return torch.bfloat16
                        return torch.float16

                    def _maybe_checkpoint(fn, *a, **k):
                        # Only checkpoint for block modules (ckpt_wrap_types) and when policy says so.
                        if not self._ckpt_on:
                            return fn(*a, **k)
                        if not isinstance(mod, self.cfg.ckpt_wrap_types):
                            return fn(*a, **k)
                        if st.ckpt_blacklisted:
                            return fn(*a, **k)
                        # gate by last observed act size; if unknown yet, skip this step
                        if st.last_act_bytes < self.cfg.min_activation_bytes_to_ckpt:
                            return fn(*a, **k)

                        preserve = _needs_rng_state(mod)

                        # local peak measurement (CUDA only) to detect harmful modules
                        pre_peak = torch.cuda.max_memory_allocated() if _has_cuda() else 0
                        out = checkpoint(fn, *a, use_reentrant=False, preserve_rng_state=preserve, **k)
                        if _has_cuda():
                            post_peak = torch.cuda.max_memory_allocated()
                            if (post_peak - pre_peak) > self.cfg.ckpt_harmful_delta_bytes:
                                st.harmful_count += 1
                                if st.harmful_count >= self.cfg.ckpt_harmful_patience:
                                    st.ckpt_blacklisted = True
                            else:
                                st.harmful_count = 0
                        return out

                    def fwd(*args, **kwargs):
                        # precision (autocast) selection
                        ac_enabled = _local_autocast_enabled()
                        dtype = _local_dtype()

                        def _body(*a, **k):
                            return mod._dpcs_orig_forward(*a, **k)  # type: ignore[attr-defined]

                        if ac_enabled:
                            with torch.autocast(device_type=self.device_type, dtype=dtype, enabled=True):
                                out = _maybe_checkpoint(_body, *args, **kwargs)
                        else:
                            out = _maybe_checkpoint(_body, *args, **kwargs)

                        # update activation size after the fact for next-step gating
                        try:
                            self._registry[mod].last_act_bytes = _tensor_bytes(out)
                        except Exception:
                            pass
                        return out

                    return fwd

                m.forward = make_fwd(m)  # type: ignore[method-assign]
        return model

    # logging helpers
    def precision_mix(self) -> dict:
        if not self._registry:
            return {}
        return dict(Counter(st.mode for st in self._registry.values()))

    def modes_summary(self) -> Dict[str, int]:
        return {k: v for k, v in self.precision_mix().items()}

    def set_log_jsonl(self, path: str):
        import json, os
        if not path:
            self._log_cb = None
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        f = open(path, "a", buffering=1)
        def _emit(payload: dict):
            try:
                f.write(json.dumps(payload) + "\n")
            except Exception:
                pass
        self._log_cb = _emit

    def _emit_log(self, payload: dict):
        cb = self._log_cb
        if cb is not None:
            try:
                cb(payload)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Step lifecycle
    # ------------------------------------------------------------------
    def start_step(self) -> None:
        self._step += 1

    def collect_signals(self, loss: torch.Tensor, model: nn.Module):
        # Stats collection is independent of precision toggle
        beta = self.cfg.ema_beta
        for mod, st in self._registry.items():
            try:
                g2 = 0.0
                cnt = 0
                for p in mod.parameters(recurse=False):
                    if p.grad is None:
                        continue
                    g2 += float(p.grad.detach().to(dtype=torch.float32).pow(2).sum().item())
                    cnt += p.numel()
                if cnt > 0:
                    g2 = g2 / cnt
                    st.grad_l2_ema = _ema(st.grad_l2_ema, g2, beta)
            except Exception:
                pass

    def end_step(self, optim: torch.optim.Optimizer, scaler: Optional[torch.amp.GradScaler] = None) -> None:
        # global overflow hint via GradScaler scale drop
        force_fp32_global = False
        if scaler is not None and hasattr(scaler, "get_scale"):
            try:
                cur_scale = float(scaler.get_scale())
                if getattr(self, "_last_scale", None) is not None and cur_scale < float(self._last_scale):
                    force_fp32_global = True
                self._last_scale = cur_scale
            except Exception:
                pass

        for m, st in self._registry.items():
            # cooldown handling
            if st.cool > 0:
                st.mode = "fp32"
                if st.just_set_cooldown:
                    st.just_set_cooldown = False
                else:
                    st.cool = max(st.cool - 1, 0)
                continue

            if force_fp32_global or st.pending_overflow:
                st.mode = "fp32"
                st.cool = max(st.cool, self.cfg.cooldown_steps)
                st.just_set_cooldown = True
                st.pending_overflow = False
                continue

            # default policy: prefer low precision when grads are small or curvature large (stub curv)
            if st.grad_l2_ema is None:
                next_mode = "fp16"
            else:
                if st.grad_l2_ema < self.cfg.epsilon_g:
                    next_mode = "fp16"
                else:
                    next_mode = "fp32"
            st.mode = next_mode

        # optional compact log
        try:
            mix = self.modes_summary()
        except Exception:
            mix = {}
        self._emit_log({
            "step": int(self._step),
            "mix": mix,
        })

        self._freeze_active = False
        self._mode_freeze.clear()

    # convenience
    def on_log(self, fn: Callable[[dict], None]) -> None:
        self._log_cb = fn

    def is_checkpointing(self) -> bool:
        return self._ckpt_on
