from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, Any, Type, List, Set
from collections import Counter
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
    grad_l2_ema_prev: Optional[float] = None
    gvar_ema: Optional[float] = None
    curv_ema: Optional[float] = None

    # per-step samples
    last_var_step: Optional[float] = None
    last_curv_step: Optional[float] = None

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

    # warm-up autotune for precision thresholds
    autotune_precision: bool = False
    autotune_warmup_steps: int = 0
    autotune_gvar_percentile: float = 0.5   # e.g., median
    autotune_curv_percentile: float = 0.5
    autotune_min_eps: float = 1e-12
    autotune_min_kappa: float = 1e-12

    # checkpointing policy: block gating and rank selection
    min_activation_bytes_to_ckpt: int = 16 << 20   # 16 MiB gate
    ckpt_harmful_delta_bytes: int = 8 << 20        # local peak rises > 8 MiB while ckpt'ing this block
    ckpt_harmful_patience: int = 3                 # consecutive steps before blacklisting

    # NEW: rank-based selection (top-K by activation bytes)
    ckpt_enable_topk: bool = True
    ckpt_topk_frac: float = 0.3                    # checkpoint top 30% biggest blocks
    ckpt_min_candidates: int = 1                   # at least this many when any exist
    ckpt_max_blocks: Optional[int] = None          # hard cap; None = no cap

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

        # warm-up autotune pools
        self._warm_enabled: bool = bool(self.cfg.autotune_precision and self.cfg.autotune_warmup_steps > 0)
        self._warm_done: bool = False
        self._warm_seen_steps: int = 0
        self._warm_gvars: List[float] = []
        self._warm_curvs: List[float] = []

        # per-step checkpoint selection (top-K)
        self._ckpt_selected: Set[nn.Module] = set()

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
                        # Checkpoint only if selected via top-K for this step
                        if not self._ckpt_on:
                            return fn(*a, **k)
                        if not isinstance(mod, self.cfg.ckpt_wrap_types):
                            return fn(*a, **k)
                        if st.ckpt_blacklisted:
                            return fn(*a, **k)
                        if self.cfg.ckpt_enable_topk:
                            if mod not in self._ckpt_selected:
                                return fn(*a, **k)
                        else:
                            # legacy absolute threshold path
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
        # Build the per-step top-K selection set for checkpointing
        self._ckpt_selected.clear()
        if not self._ckpt_on or not self.cfg.ckpt_enable_topk:
            return
        # collect eligible candidates
        cands: List[Tuple[int, nn.Module]] = []
        for m, st in self._registry.items():
            if not isinstance(m, self.cfg.ckpt_wrap_types):
                continue
            if st.ckpt_blacklisted:
                continue
            if st.last_act_bytes < self.cfg.min_activation_bytes_to_ckpt:
                continue
            cands.append((st.last_act_bytes, m))
        if not cands:
            return
        cands.sort(key=lambda t: t[0], reverse=True)
        # compute K
        K = len(cands)
        if self.cfg.ckpt_topk_frac > 0:
            K = max(int(math.ceil(self.cfg.ckpt_topk_frac * len(cands))), int(self.cfg.ckpt_min_candidates))
        if self.cfg.ckpt_max_blocks is not None:
            K = min(K, int(self.cfg.ckpt_max_blocks))
        K = max(0, min(K, len(cands)))
        selected = [m for _, m in cands[:K]]
        self._ckpt_selected = set(selected)

    def _variance_and_l2_mean(self, mod: nn.Module) -> Optional[Tuple[float, float]]:
        """Return (population variance, mean(g^2)) across all param grads in module, or None if no grads."""
        total_elems = 0
        s = 0.0
        ss = 0.0
        for p in mod.parameters(recurse=False):
            g = p.grad
            if g is None:
                continue
            g32 = g.detach().to(torch.float32)
            total_elems += g32.numel()
            s  += float(g32.sum().item())
            ss += float((g32 * g32).sum().item())
        if total_elems == 0:
            return None
        mean = s / total_elems
        var = max(ss / total_elems - mean * mean, 0.0)  # population variance
        l2_mean = ss / total_elems                       # E[g^2]
        return var, l2_mean

    def collect_signals(self, loss: torch.Tensor, model: nn.Module):
        # Stats collection is independent of precision toggle
        beta = self.cfg.ema_beta
        # per-step pools for autotune
        step_gvars: List[float] = []
        step_curvs: List[float] = []

        for mod, st in self._registry.items():
            try:
                out = self._variance_and_l2_mean(mod)
                if out is not None:
                    var, l2_mean = out
                    # per-step variance sample
                    st.last_var_step = var
                    st.gvar_ema = _ema(st.gvar_ema, var, beta)

                    # grad-L2 EMA + curvature proxy (ensure defined curvature on first step)
                    if st.grad_l2_ema is None:
                        st.grad_l2_ema = l2_mean
                        st.grad_l2_ema_prev = l2_mean
                        st.last_curv_step = 0.0
                    else:
                        prev = st.grad_l2_ema
                        st.grad_l2_ema = _ema(prev, l2_mean, beta)
                        eps = 1e-12
                        st.last_curv_step = abs(math.log(st.grad_l2_ema + eps) - math.log(prev + eps))

                    # accumulate for autotune pools (now always defined)
                    step_gvars.append(st.last_var_step)
                    step_curvs.append(st.last_curv_step)
            except Exception:
                pass

        # warm-up autotune after finishing this step
        if self._warm_enabled and not self._warm_done:
            self._warm_seen_steps += 1
            self._warm_gvars.extend(step_gvars)
            self._warm_curvs.extend(step_curvs)
            if self._warm_seen_steps >= int(self.cfg.autotune_warmup_steps):
                # compute percentiles (torch.quantile expects tensor)
                try:
                    if len(self._warm_gvars) > 0:
                        t_g = torch.tensor(self._warm_gvars, dtype=torch.float64)
                        qg = torch.tensor(float(self.cfg.autotune_gvar_percentile), dtype=torch.float64)
                        eps_ref = float(torch.quantile(t_g, qg).item())
                    else:
                        eps_ref = float(self.cfg.autotune_min_eps)
                except Exception:
                    eps_ref = float(self.cfg.autotune_min_eps)
                try:
                    if len(self._warm_curvs) > 0:
                        t_c = torch.tensor(self._warm_curvs, dtype=torch.float64)
                        qc = torch.tensor(float(self.cfg.autotune_curv_percentile), dtype=torch.float64)
                        kap_ref = float(torch.quantile(t_c, qc).item())
                    else:
                        kap_ref = float(self.cfg.autotune_min_kappa)
                except Exception:
                    kap_ref = float(self.cfg.autotune_min_kappa)

                self.cfg.epsilon_g = max(eps_ref, float(self.cfg.autotune_min_eps))
                self.cfg.kappa     = max(kap_ref, float(self.cfg.autotune_min_kappa))
                self._warm_done = True

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

            # default policy: prefer low precision when grads are small (curv optional)
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
        if self._log_cb:
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
