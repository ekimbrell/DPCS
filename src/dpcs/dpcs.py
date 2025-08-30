from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, Any, Type, List, Set, Iterable
from collections import Counter
from contextlib import contextmanager, nullcontext
import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

# --- Optional dependencies ----------------------------------------------------
try:
    import transformer_engine.pytorch as te  # Optional FP8 backend (not exercised here)
    _TE_AVAILABLE = True
except Exception:
    _TE_AVAILABLE = False

# SDPA kernel context (PyTorch >= 2.1). Provide a safe fallback for older versions.
try:
    from torch.nn.attention import sdpa_kernel, SDPBackend  # PyTorch 2.1+
    _SDPA_AVAILABLE = True
except Exception:  # pragma: no cover - fallback path
    _SDPA_AVAILABLE = False

    @contextmanager
    def sdpa_kernel(*_args, **_kwargs):  # type: ignore[misc]
        yield

    class SDPBackend:  # type: ignore[override]
        MATH = "MATH"
        EFFICIENT_ATTENTION = "EFFICIENT_ATTENTION"
        FLASH_ATTENTION = "FLASH_ATTENTION"

# --- Small helpers ------------------------------------------------------------

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


# --- Per-module state ---------------------------------------------------------

@dataclass
class _ModuleState:
    # precision policy
    mode: str = "fp16"            # 'fp32' | 'fp16' | 'fp8'
    cool: int = 0                 # cooldown steps after overflow

    # running stats for precision/autotune
    grad_l2_ema: Optional[float] = None
    grad_l2_ema_prev: Optional[float] = None
    gvar_ema: Optional[float] = None

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


# --- Config -------------------------------------------------------------------

@dataclass
class DPCSConfig:
    # device / wrapping
    device_type: str = "cuda"

    # precision scheduler knobs
    enable_precision: bool = True
    epsilon_g: float = 1e-3
    kappa: float = 5.0
    cooldown_steps: int = 3
    ema_beta: float = 0.9

    # warm-up autotune for precision thresholds
    autotune_precision: bool = False
    autotune_warmup_steps: int = 0
    autotune_gvar_percentile: float = 0.5
    autotune_curv_percentile: float = 0.5
    autotune_min_eps: float = 1e-12
    autotune_min_kappa: float = 1e-12

    # checkpointing policy (top-K by activation size)
    min_activation_bytes_to_ckpt: int = 16 << 20   # 16 MiB gate
    ckpt_harmful_delta_bytes: int = 8 << 20
    ckpt_harmful_patience: int = 3

    ckpt_enable_topk: bool = True
    ckpt_topk_frac: float = 0.3
    ckpt_min_candidates: int = 1
    ckpt_max_blocks: Optional[int] = None

    # memory-aware precision policy
    low_headroom_frac: float = 0.12  # e.g., prefer lower precision below ~12% free VRAM

    # checkpoint determinism check: "default" compares (shape,dtype,device); "none" disables the check
    checkpoint_determinism_check: str = "none"

    # which module types we precision-wrap vs ckpt-wrap
    wrap_types: Tuple[Type[nn.Module], ...] = (nn.Linear, nn.TransformerEncoderLayer)
    ckpt_wrap_types: Tuple[Type[nn.Module], ...] = (nn.Sequential, nn.TransformerEncoderLayer)

    # --- NEW: SDPA & checkpoint robustness knobs ---
    force_sdpa_in_blocks: bool = True
    # Accept either SDPBackend enums or their names ('MATH', 'EFFICIENT_ATTENTION', 'FLASH_ATTENTION')
    sdpa_backends: Tuple[Any, ...] = (SDPBackend.MATH,)  # type: ignore[name-defined]


# --- Main scheduler -----------------------------------------------------------

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

        # precision toggle determines autocast use; stats are always collected
        self._prec_on = bool(self.cfg.enable_precision)
        self._low_headroom: bool = False

        # warm-up autotune pools
        self._warm_enabled: bool = bool(self.cfg.autotune_precision and self.cfg.autotune_warmup_steps > 0)
        self._warm_done: bool = False
        self._warm_seen_steps: int = 0
        self._warm_gvars: List[float] = []
        self._warm_curvs: List[float] = []

        # per-step checkpoint selection (top-K)
        self._ckpt_selected: Set[nn.Module] = set()

        # normalize sdpa backends once
        self._sdpa_backends = self._normalize_sdpa_backends(self.cfg.sdpa_backends)

    # --- SDPA backends normalization -------------------------------------
    def _normalize_sdpa_backends(self, backends: Iterable[Any]) -> Tuple[Any, ...]:
        norm: List[Any] = []
        for b in backends:
            if _SDPA_AVAILABLE and isinstance(b, SDPBackend):  # already enum
                norm.append(b)
            elif _SDPA_AVAILABLE and isinstance(b, str):
                name = b.upper()
                if hasattr(SDPBackend, name):
                    norm.append(getattr(SDPBackend, name))
            else:
                # fallback path or unknown: keep as-is; sdpa_kernel will be a no-op
                norm.append(b)
        return tuple(norm)

    # --- AMP helpers--------------------------------------
    def amp_dtype(self) -> torch.dtype:
        """Preferred autocast dtype for this device (bf16 if supported, else fp16)."""
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16


    def amp_uses_grad_scaler(self) -> bool:
        """Whether the training loop should use GradScaler for the current AMP dtype."""
        return self.amp_dtype() == torch.float16

    # --- Public API -------------------------------------------------------
    def wrap(self, model: nn.Module) -> nn.Module:
        for m in model.modules():
            if isinstance(m, self.cfg.wrap_types) and not hasattr(m, "_dpcs_orig_forward"):
                self._registry[m] = _ModuleState(mode="fp16")
                m._dpcs_orig_forward = m.forward  # type: ignore[attr-defined]

                def make_fwd(mod: nn.Module):
                    st = self._registry[mod]

                    def _needs_rng_state(mod_: nn.Module) -> bool:
                        for sub in mod_.modules():
                            if isinstance(sub, (nn.Dropout, nn.AlphaDropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)) and sub.p > 0 and sub.training:
                                return True
                            if hasattr(sub, "dropout") and isinstance(getattr(sub, "dropout"), float) and getattr(sub, "dropout") > 0 and sub.training:
                                return True
                        return False

                    def _local_autocast_enabled() -> bool:
                        if not self._prec_on:
                            return False
                        mode = self._mode_freeze.get(mod, st.mode)
                        return mode in ("fp16", "fp8")  # fp8 path would be provided by a backend adapter

                    def _local_dtype():
                        # delegate to class-wide AMP dtype policy (bf16 preferred)
                        return self.amp_dtype()

                    def _ensure_topk_selected_if_needed():
                        # Lazy top-K selection so users who forget start_step() still checkpoint
                        if not self._ckpt_on or not self.cfg.ckpt_enable_topk or self._ckpt_selected:
                            return
                        cands: List[Tuple[int, nn.Module]] = []
                        for m2, st2 in self._registry.items():
                            if not isinstance(m2, self.cfg.ckpt_wrap_types):
                                continue
                            if st2.ckpt_blacklisted:
                                continue
                            if st2.last_act_bytes < self.cfg.min_activation_bytes_to_ckpt:
                                continue
                            cands.append((st2.last_act_bytes, m2))
                        if not cands:
                            return
                        cands.sort(key=lambda t: t[0], reverse=True)
                        K = max(int(math.ceil(self.cfg.ckpt_topk_frac * len(cands))), int(self.cfg.ckpt_min_candidates))
                        if self.cfg.ckpt_max_blocks is not None:
                            K = min(K, int(self.cfg.ckpt_max_blocks))
                        self._ckpt_selected = set(m for _, m in cands[:K])

                    def _maybe_checkpoint(fn, *a, **k):
                        if not self._ckpt_on:
                            return fn(*a, **k)
                        if not isinstance(mod, self.cfg.ckpt_wrap_types):
                            return fn(*a, **k)
                        if st.ckpt_blacklisted:
                            return fn(*a, **k)

                        _ensure_topk_selected_if_needed()
                        if self.cfg.ckpt_enable_topk:
                            if mod not in self._ckpt_selected:
                                return fn(*a, **k)
                        else:
                            if st.last_act_bytes < self.cfg.min_activation_bytes_to_ckpt:
                                return fn(*a, **k)

                        preserve = _needs_rng_state(mod)
                        pre_peak = torch.cuda.memory.max_memory_allocated() if _has_cuda() else 0
                        out = torch.utils.checkpoint.checkpoint(_body, *a, use_reentrant=False, determinism_check=self.cfg.checkpoint_determinism_check, **k)
                        if _has_cuda():
                            post_peak = torch.cuda.memory.max_memory_allocated()
                            if (post_peak - pre_peak) > self.cfg.ckpt_harmful_delta_bytes:
                                st.harmful_count += 1
                                if st.harmful_count >= self.cfg.ckpt_harmful_patience:
                                    st.ckpt_blacklisted = True
                            else:
                                st.harmful_count = 0
                        return out

                    def _body(*a, **k):
                        return mod._dpcs_orig_forward(*a, **k)  # type: ignore[attr-defined]

                    def fwd(*args, **kwargs):
                        ac_enabled = _local_autocast_enabled()
                        dtype = _local_dtype()

                        # SDPA robustness: if this is a Transformer block and the knob is on,
                        # enter a stable sdpa_kernel context so comparisons are apples-to-apples.
                        is_tx = isinstance(mod, nn.TransformerEncoderLayer)
                        sdpa_ctx = sdpa_kernel(self._sdpa_backends) if (self.cfg.force_sdpa_in_blocks and is_tx) else nullcontext()

                        if ac_enabled:
                            with torch.amp.autocast(self.device_type, dtype=dtype, enabled=True):
                                with sdpa_ctx:
                                    out = _maybe_checkpoint(_body, *args, **kwargs)
                        else:
                            with sdpa_ctx:
                                out = _maybe_checkpoint(_body, *args, **kwargs)

                        # update activation size for next-step gating
                        try:
                            self._registry[mod].last_act_bytes = _tensor_bytes(out)
                        except Exception:
                            pass
                        return out

                    return fwd

                m.forward = make_fwd(m)  # type: ignore[method-assign]
        return model

    # --- Logging helpers --------------------------------------------------
    # Precision control 
    def set_precision_enabled(self, on: bool = True) -> None:
        """
        Toggle global mixed precision. When False, DPCS runs modules in fp32.
        """
        self._prec_on = bool(on)

    def get_amp_config(self) -> Tuple[str, torch.dtype, bool]:
        """
        Returns (device_type, dtype, enabled) tuple suitable for torch.amp.autocast.
        Example usage:
            device_type, dtype, enabled = self.get_amp_config()
            # with torch.amp.autocast(device_type, dtype=dtype, enabled=enabled): ...
        """
        return (self.device_type, self.amp_dtype(), self._prec_on)

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

    # --- VRAM headroom helpers ---------------------------------------------
    def _mem_get_info(self) -> Optional[Tuple[int, int]]:
        """
        Returns (free_bytes, total_bytes) for the current CUDA device, or None if unavailable.
        Prefers torch.cuda.memory.mem_get_info(); falls back to torch.cuda.mem_get_info().
        """
        if not torch.cuda.is_available():
            return None
        try:
            free, total = torch.cuda.memory.mem_get_info()  # PyTorch docs-recommended API
        except Exception:
            try:
                free, total = torch.cuda.mem_get_info()      # older alias
            except Exception:
                return None
        return int(free), int(total)

    def vram_headroom(self) -> Optional[float]:
        """
        Returns free/total VRAM fraction in [0,1], or None if not on CUDA.
        """
        info = self._mem_get_info()
        if info is None:
            return None
        free, total = info
        return (float(free) / float(total)) if total else None

    # --- Step lifecycle ---------------------------------------------------
    def start_step(self) -> None:
        self._step += 1
        # Reset CUDA peak memory stats so per-step peaks are comparable.
        # (Per docs, reset_peak_memory_stats() scopes max_memory_allocated() to the period after reset.)
        # Fallback to torch.cuda.reset_peak_memory_stats() for older installs.
        if torch.cuda.is_available():
            try:
                torch.cuda.memory.reset_peak_memory_stats()
            except Exception:
                try:
                    torch.cuda.reset_peak_memory_stats()
                except Exception:
                    pass
        # Measure live VRAM headroom (free/total) for this step and cache a flag.
        self._low_headroom = False
        ratio = self.vram_headroom()
        if ratio is not None and ratio < float(self.cfg.low_headroom_frac):
            self._low_headroom = True


        # Build the per-step top-K selection set for checkpointing
        self._ckpt_selected.clear()
        if not self._ckpt_on or not self.cfg.ckpt_enable_topk:
            return
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

        # Start from configured fraction, but if headroom is low, bias toward more checkpointing.
        total_candidates = len(cands)  # include ancestors (e.g., outer Sequential) for target-K math
        frac = float(self.cfg.ckpt_topk_frac)
        if getattr(self, "_low_headroom", False):
            # Increase fraction under pressure, but never above 1.0
            frac = min(1.0, max(frac, frac * 1.5))
        if frac > 0:
            K_target = max(int(math.ceil(frac * total_candidates)), int(self.cfg.ckpt_min_candidates))
        else:
            K_target = 0
        if self.cfg.ckpt_max_blocks is not None:
            K_target = min(K_target, int(self.cfg.ckpt_max_blocks))
        K_target = max(0, K_target)

        # Leaf-only filter: keep modules that do NOT contain other ckpt-wrap types as descendants.
        # This avoids selecting both an outer container and one of its inner blocks.
        def _has_ckpt_descendant(m: nn.Module) -> bool:
            for ch in m.modules():
                if ch is m:
                    continue
                if isinstance(ch, self.cfg.ckpt_wrap_types):
                    return True
            return False

        leaf_cands = [(bytes_, m) for (bytes_, m) in cands if not _has_ckpt_descendant(m)]

        # Cap K by the number of leaf candidates (no nesting conflicts among leaves).
        K = min(K_target, len(leaf_cands))

        # If nothing left (e.g., all were non-leaf), fall back to original cands but avoid conflicts.
        effective_cands = leaf_cands if K > 0 else cands

        # Select up to K modules while avoiding ancestor/descendant conflicts as a safety net.
        selected: list[nn.Module] = []
        def _conflicts(m: nn.Module) -> bool:
            for s in selected:
                if s is m:
                    return True
                for ch in s.modules():
                    if ch is m and ch is not s:
                        return True
                for ch in m.modules():
                    if ch is s and ch is not m:
                        return True
            return False

        for _, m in effective_cands:
            if len(selected) >= K:
                break
            if _conflicts(m):
                continue
            selected.append(m)

        self._ckpt_selected = set(selected)




    # --- Stats collection -------------------------------------------------
    def _variance_and_l2_mean(self, mod: nn.Module) -> Optional[Tuple[float, float]]:
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
        beta = self.cfg.ema_beta
        step_gvars: List[float] = []
        step_curvs: List[float] = []

        for mod, st in self._registry.items():
            try:
                out = self._variance_and_l2_mean(mod)
                if out is not None:
                    var, l2_mean = out
                    st.last_var_step = var
                    st.gvar_ema = _ema(st.gvar_ema, var, beta)

                    if st.grad_l2_ema is None:
                        st.grad_l2_ema = l2_mean
                        st.grad_l2_ema_prev = l2_mean
                        st.last_curv_step = 0.0
                    else:
                        prev = st.grad_l2_ema
                        st.grad_l2_ema = _ema(prev, l2_mean, beta)
                        eps = 1e-12
                        st.last_curv_step = abs(math.log(st.grad_l2_ema + eps) - math.log(prev + eps))

                    step_gvars.append(st.last_var_step)
                    step_curvs.append(st.last_curv_step)
            except Exception:
                pass

        if self._warm_enabled and not self._warm_done:
            self._warm_seen_steps += 1
            self._warm_gvars.extend(step_gvars)
            self._warm_curvs.extend(step_curvs)
            if self._warm_seen_steps >= int(self.cfg.autotune_warmup_steps):
                try:
                    eps_ref = float(torch.quantile(torch.tensor(self._warm_gvars, dtype=torch.float64), torch.tensor(float(self.cfg.autotune_gvar_percentile), dtype=torch.float64)).item()) if self._warm_gvars else float(self.cfg.autotune_min_eps)
                except Exception:
                    eps_ref = float(self.cfg.autotune_min_eps)
                try:
                    kap_ref = float(torch.quantile(torch.tensor(self._warm_curvs, dtype=torch.float64), torch.tensor(float(self.cfg.autotune_curv_percentile), dtype=torch.float64)).item()) if self._warm_curvs else float(self.cfg.autotune_min_kappa)
                except Exception:
                    kap_ref = float(self.cfg.autotune_min_kappa)

                self.cfg.epsilon_g = max(eps_ref, float(self.cfg.autotune_min_eps))
                self.cfg.kappa     = max(kap_ref, float(self.cfg.autotune_min_kappa))
                self._warm_done = True

    # --- Step end ---------------------------------------------------------
    def end_step(self, optim: torch.optim.Optimizer, scaler: Optional[torch.amp.GradScaler] = None) -> None:
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

            # Memory-aware precision policy with a strict FP32 override.
            # If global precision is disabled, force fp32 regardless of headroom.
            if not self._prec_on:
                next_mode = "fp32"
            else:
                if getattr(self, "_low_headroom", False):
                    # prefer AMP under pressure (bf16 preferred by amp_dtype)
                    next_mode = "fp16"
                else:
                    # comfortable headroom -> fp32
                    next_mode = "fp32"
            st.mode = next_mode


        # optional compact log
        if self._log_cb:
            try:
                self._log_cb({"step": int(self._step), "mix": self.modes_summary()})
            except Exception:
                pass

        self._freeze_active = False
        self._mode_freeze.clear()

    # --- Convenience ------------------------------------------------------
    def on_log(self, fn: Callable[[dict], None]) -> None:
        self._log_cb = fn

    def enable_checkpointing(self, on: bool = True) -> None:
        """
        Enable/disable activation checkpointing globally.
        """
        self._ckpt_on = bool(on)

    def set_checkpoint_fraction(self, frac: float) -> None:
        """
        Set fraction of checkpoint-eligible modules to select (0.0–1.0).
        """
        frac = float(frac)
        self.cfg.ckpt_topk_frac = max(0.0, min(frac, 1.0))

    def set_checkpoint_determinism_check(self, mode: str = "none") -> None:
        """
        Set checkpoint determinism check mode: "default" or "none".
        """
        if mode not in ("default", "none"):
            raise ValueError(f"invalid determinism_check={mode}")
        self.cfg.checkpoint_determinism_check = mode

    def is_checkpointing(self) -> bool:
        return self._ckpt_on
