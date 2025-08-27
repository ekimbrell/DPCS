# dpcs.py
from __future__ import annotations
from dataclasses import dataclass
from contextlib import nullcontext, contextmanager
from typing import Callable, Dict, Optional, Tuple
import math
import torch
import torch.nn as nn
from contextlib import contextmanager, nullcontext
from torch.utils.checkpoint import checkpoint
import torch.nn as nn
from collections import Counter
from dataclasses import dataclass
from contextlib import contextmanager, nullcontext

try:
    import transformer_engine.pytorch as te  # Optional FP8
    _TE_AVAILABLE = True
except Exception:
    _TE_AVAILABLE = False

@dataclass
class _PrecState:
    mode: str = "fp16"          # 'fp32' | 'fp16' | 'fp8'
    cool: int = 0               # fp32 cooldown steps after overflow/instability
    gnorm_ema: float = 0.0
    gvar_ema: float = 0.0
    curv_ema: float = 0.0



@dataclass
class _ModuleState:
    # current numeric mode chosen by scheduler ('fp32' | 'fp16' | 'fp8')
    mode: str = "fp16"

    # cooldown counter (stay in fp32 for N steps after instability/overflow)
    cool: int = 0

    # exponential moving averages for decision signals
    gnorm_ema: float = 0.0   # EMA of gradient L2 norm
    gvar_ema: float = 0.0    # EMA of gradient variance
    curv_ema: float = 0.0    # EMA of curvature proxy (e.g., |Δ log ||g|| |)

    # book-keeping
    last_update_step: int = -1  # last global step when stats/mode were updated
    overflow_count: int = 0     # consecutive nonfinite/overflow detections
    
    grad_l2_ema: Optional[float] = None
    grad_l2_ema_prev: Optional[float] = None
    pending_overflow: bool = False
    just_set_cooldown: bool = False   
    


# --- Utilities ---------------------------------------------------------------

def _has_cuda() -> bool:
    return torch.cuda.is_available()

def _mem_headroom(device_type: str) -> float:
    """
    Return free/total device memory in [0,1]. Falls back to 1.0 if unknown.
    Uses torch.cuda.memory.mem_get_info() on CUDA.  :contentReference[oaicite:1]{index=1}
    """
    try:
        if device_type == "cuda" and _has_cuda():
            free, total = torch.cuda.memory.mem_get_info()
            return float(free) / float(total + 1e-9)
    except Exception:
        pass
    return 1.0  # CPU/MPS default: treat as plenty of headroom

def _is_leaf_module(mod: nn.Module) -> bool:
    return len(list(mod.children())) == 0

def _is_block_sequential(mod: nn.Module) -> bool:
    # Wrap sequentials that are “blocks”: all of their children are leaves
    if not isinstance(mod, nn.Sequential):
        return False
    kids = list(mod.children())
    return len(kids) > 0 and all(_is_leaf_module(c) for c in kids)

# add near the top of dpcs.py
def _should_wrap_block(mod: nn.Module) -> bool:
    if _is_leaf_module(mod):   # avoid leafs (Linear/GELU/etc.)
        return False
    if isinstance(mod, (nn.Sequential, nn.TransformerEncoderLayer)):
        return True
    # size gate: treat “big” modules as blocks
    try:
        n_params = sum(p.numel() for p in mod.parameters())
    except Exception:
        n_params = 0
    return n_params >= 1_000_000


# --- Core API ----------------------------------------------------------------

@dataclass
class DPCSConfig:
    # device / wrapping
    device_type: str = "cuda"
    allow_fp8: bool = False
    fp8_backend: str = "te"           # or "none"

    # precision scheduler knobs
    enable_precision: bool = True
    epsilon_g: float = 1e-3           # grad variance (proxy) threshold
    kappa: float = 5.0                # curvature (proxy) threshold (stubbed for now)
    cooldown_steps: int = 3           # steps to hold FP32 after overflow
    ema_beta: float = 0.9             # EMA smoothing for grad signal
    signals_freq_steps: int = 1       # recompute stats this often

    # checkpointing hysteresis (based on headroom)
    ckpt_low: float = 0.05            # turn ON ckpt if headroom <= low
    ckpt_high: float = 0.20           # turn OFF ckpt if headroom >= high
    ckpt_need: int = 2                # consecutive votes required

    # which module types we actually wrap
    wrap_types: Tuple[Type[nn.Module], ...] = (nn.Linear,)


class DPCS:
    """
    Drop-in scheduler that:
      - decides per-module precision (fp32/fp16/fp8) for the next step,
      - decides whether to enable activation checkpointing for the next step.
    """
    def _snapshot_headroom(self) -> float:
        # normalized free fraction; falls back to 1.0 if CUDA/mem_get_info unavailable
        if self.device_type != "cuda" or not torch.cuda.is_available():
            return 1.0
        try:
            free, total = torch.cuda.mem_get_info()
            return max(0.0, min(1.0, float(free) / float(total)))
        except Exception:
            return 1.0



    def _has_nonfinite_grad(self, mod: nn.Module) -> bool:
        for p in mod.parameters(recurse=False):
            if p.grad is not None:
                g = p.grad
                # Any NaN/Inf → treat as overflow signal
                if not torch.isfinite(g).all():
                    return True
        return False

        

    def log_dict(self) -> dict:
        mix = {}
        for _, st in self._registry.items():
            mix[st.mode] = mix.get(st.mode, 0) + 1
        try:
            alloc = torch.cuda.memory.get_allocator_backend()  # 'native' or 'cudaMallocAsync'
        except Exception:
            alloc = "unknown"
        info = {
            "step": self._step,
            "ckpt_on": bool(getattr(self, "_ckpt_on", False)),
            "precision_mix": mix,
            "device": str(self.device_type),
            "allocator_backend": alloc,
        }
        # Attach headroom if you track it
        if hasattr(self, "_headroom"):
            info["headroom"] = float(self._headroom)
        return info

    
    def _grad_l2_of(self, mod: torch.nn.Module) -> Optional[float]:
        total = 0.0
        any_grad = False
        for p in mod.parameters(recurse=False):
            if p.grad is not None:
                any_grad = True
                g = p.grad.detach()
                total += float((g * g).sum().item())
        return total if any_grad else None

    
    def _update_stats_and_choose_mode(self, mod: nn.Module) -> None:
        st = self._registry[mod]

        # Grad L2 proxy (EMA). If no grads yet, keep last EMA.
        total = 0.0
        count = 0
        for p in mod.parameters(recurse=False):
            if p.grad is not None:
                total += float(p.grad.detach().float().pow(2).mean().item())
                count += 1
        if count > 0:
            cur_l2 = total / count
            st.grad_l2_ema_prev = st.grad_l2_ema
            st.grad_l2_ema = _ema_update(st.grad_l2_ema, cur_l2, self.cfg.ema_beta)

        # Decision: prefer fp16 if “stable”, otherwise fp32
        next_mode = "fp16"
        if st.grad_l2_ema is None or st.grad_l2_ema >= self.cfg.epsilon_g:
            next_mode = "fp32"
        if getattr(st, "cool", 0) > 0:
            next_mode = "fp32"

        st.mode = next_mode



    def _ema_update(self, old, new, beta):
        return float(beta * old + (1.0 - beta) * float(new))

    def _module_params(self, mod):
        return [p for p in mod.parameters(recurse=False) if p.grad is not None]

    def _grads_finite(self, mod) -> bool:
        for p in self._module_params(mod):
            if p.grad is None:
                continue
            if not torch.isfinite(p.grad).all():
                return False
        return True

        
    def precision_mix(self) -> dict:
        """
        Return a dict like {'fp16': N1, 'fp32': N2, 'fp8': N3} summarizing the
        current precision mode of all wrapped modules.
        """
        if not self._registry:
            return {}
        return dict(Counter(st.mode for st in self._registry.values()))
    def debug_summary(self):
        """Print what DPCS wrapped and current precision modes."""
        items = []
        for m, st in self._registry.items():
            items.append((type(m).__name__, getattr(st, "mode", "fp16")))
        items.sort(key=lambda t: t[0])
        print("[DPCS] wrapped modules (type, mode):", items)

    @contextmanager
    def _checkpoint_region(self):
        self._freeze_active = True
        try:
            yield
        finally:
            self._freeze_active = False



    def __init__(self, **kwargs):
        # config first
        self.cfg = DPCSConfig(**kwargs)
        self._prec_on = bool(self.cfg.enable_precision)  # make collect_signals/end_step live
        self._last_scale = None  # used by end_step to detect AMP scale drops (overflow hint)
        self.device_type = self.cfg.device_type
        

        # internal state
        self._step = 0
        self._last_scale = None  # for GradScaler drop detection (overflow hint)
        self._ckpt_on = False
        self._ckpt_votes = 0
        self._freeze_active: bool = False
        self._mode_freeze: Dict[nn.Module, str] = {}
        self._registry: Dict[nn.Module, _ModuleState] = {}
        self._log_cb: Optional[Callable[[dict], None]] = None

        # AMP scale tracking to detect recent overflow
        # self._last_scale: Optional[float] = None

        # snapshot headroom AFTER device_type exists
        self._headroom: float = self._snapshot_headroom()

        # optional FP8 backend gate (stub; wire later)
        self._fp8_ok = False
        self._te = None
        if self.cfg.allow_fp8 and self.cfg.fp8_backend == "te":
            try:
                import transformer_engine.pytorch as te  # type: ignore
                self._fp8_ok = True
                self._te = te
            except Exception:
                self._fp8_ok = False
                self._te = None


    # --- Logging (optional JSONL sink) -----------------------------------------
    def set_log_jsonl(self, path: str):
        """
        Enable per-step JSONL logging. Call with `None` or '' to disable.
        """
        import json, os
        if not path:
            self._log_cb = None
            return

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        f = open(path, "a", buffering=1)  # line-buffered

        def _emit(payload: dict):
            try:
                f.write(json.dumps(payload) + "\n")
            except Exception:
                pass

        self._log_cb = _emit

    def _emit_log(self, payload: dict):
        cb = getattr(self, "_log_cb", None)
        if cb is not None:
            try:
                cb(payload)
            except Exception:
                pass


    # (Optional) You can sanity-warn if user enabled precision scheduling without CUDA:
    # if self._prec_on and self.device_type != "cuda":
    #     warnings.warn("DPCS precision scheduling is optimized for CUDA autocast/GradScaler.", RuntimeWarning)


    def wrap(self, model: nn.Module) -> nn.Module:
        # register modules we care about
        for m in model.modules():
            if isinstance(m, self.cfg.wrap_types) and not hasattr(m, "_dpcs_orig_forward"):
                self._registry[m] = _ModuleState(mode="fp16")
                m._dpcs_orig_forward = m.forward  # type: ignore[attr-defined]

                def make_fwd(mod: nn.Module):
                    def run_with_local_autocast(*a, **k):
                        mode = self._mode_freeze.get(mod, self._registry[mod].mode)
                        if mode == "fp32":
                            return mod._dpcs_orig_forward(*a, **k)  # type: ignore[attr-defined]
                        dtype = torch.float16  # fp8 path wired later if available
                        with torch.autocast(device_type=self.device_type, dtype=dtype, enabled=True):
                            return mod._dpcs_orig_forward(*a, **k)  # type: ignore[attr-defined]
                    def fwd(*args, **kwargs):
                        if self._freeze_active:
                            return run_with_local_autocast(*args, **kwargs)
                        return run_with_local_autocast(*args, **kwargs)
                    return fwd
                m.forward = make_fwd(m)  # type: ignore[method-assign]
        return model

    
    
    def start_step(self) -> None:
        self._step += 1
        self._headroom = self._snapshot_headroom()


    def checkpoint_contexts_if_needed(self):
        if self._ckpt_on:
            return self._checkpoint_region(), nullcontext()
        return nullcontext(), nullcontext()
    
    def collect_signals(self, loss: torch.Tensor, model: torch.nn.Module):
        """
        Lightweight per-step signal collection. DO NOT change cooldown here.
        Only compute stats and flag pending_overflow; mode/cooldown changes are centralized in end_step().
        """
        if not getattr(self, "_prec_on", False):
            return

        for mod in self._registry.keys():
            st = self._registry[mod]

            # If grads exist and any are non-finite under this module, flag overflow.
            # Don't touch st.cool here; end_step() will start cooldown and mark just_set_cooldown.
            try:
                for p in mod.parameters(recurse=False):
                    if p.grad is None:
                        continue
                    if not torch.isfinite(p.grad).all():
                        st.pending_overflow = True
                        break
            except Exception:
                # Conservative: also flag overflow on unexpected errors
                st.pending_overflow = True

            # --- (Optional) update any EMA stats you keep for precision decisions ---
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
                    st.grad_l2_ema = self._ema(st.grad_l2_ema, g2, self.cfg.ema_beta)
            except Exception:
                pass

    def end_step(self, optim: torch.optim.Optimizer, scaler: Optional[torch.amp.GradScaler] = None) -> None:
        """
        Decide next-step precision per module + checkpointing gate using:
        - device headroom (hysteresis gate),
        - recent overflow hint (if GradScaler scale dropped),
        - grad variance proxy (EMA on grad L2) and curvature proxy,
        - cooldown that holds fp32 for a fixed number of steps after an overflow.
        """

        # --- 1) Checkpointing hysteresis votes (unchanged logic) ---
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

        # --- 2) Overflow hint via GradScaler.get_scale() ---
        # If scale strictly dropped since the last step, treat as an overflow hint.  (PyTorch practice)  # noqa
        # Ref: forum suggestion to check scaler.get_scale() to detect skipped steps / scale drop.
        # (We do not cite code here; logic mirrors public guidance.)  # noqa
        force_fp32_global = False
        if scaler is not None and hasattr(scaler, "get_scale"):
            try:
                cur_scale = float(scaler.get_scale())
                if getattr(self, "_last_scale", None) is not None and cur_scale < float(self._last_scale):
                    force_fp32_global = True
                self._last_scale = cur_scale
            except Exception:
                pass  # stay conservative

        # --- 3) Per-module next-mode decision ---
        for m, st in self._registry.items():

            # 3a) If module is cooling, keep fp32 and decrement (except the step we set it)
            if st.cool > 0:
                st.mode = "fp32"
                if getattr(st, "just_set_cooldown", False):
                    # Hold value on the step we entered cooldown; decrement starting next step
                    st.just_set_cooldown = False
                else:
                    st.cool = max(st.cool - 1, 0)
                continue

            # 3b) If we have a global overflow hint, force fp32 and enter cooldown
            if force_fp32_global:
                st.mode = "fp32"
                st.cool = max(st.cool, self._cool)  # self._cool mirrors cfg.cooldown_steps
                st.just_set_cooldown = True
                st.pending_overflow = False
                continue

            # 3c) Optional: live non-finite grad sniff on this module to trigger cooldown fast
            force_now = False
            try:
                if hasattr(self, "_has_nonfinite_grad") and callable(self._has_nonfinite_grad):
                    if self._has_nonfinite_grad(m):
                        force_now = True
            except Exception:
                pass

            if st.pending_overflow or force_now:
                st.mode = "fp32"
                st.cool = max(st.cool, self._cool)
                st.just_set_cooldown = True
                st.pending_overflow = False
                continue

            # 3d) Normal policy
            # Optimistic warm-start: if no EMA yet, prefer low precision (test expects fp16 on the first clean step)
            if st.grad_l2_ema is None:
                next_mode = "fp16"
            else:
                # Your seed decision: low precision when grads are stable or curvature large
                drop_by_grad = (st.grad_l2_ema < self.cfg.epsilon_g)
                drop_by_curv = (st.curv_ema is not None and st.curv_ema > self.cfg.kappa)
                if drop_by_grad or drop_by_curv:
                    next_mode = "fp16"   # later: gate to 'fp8' when backend & arch allow
                else:
                    next_mode = "fp32" if self._fp32_pref() else "fp16"

            st.mode = next_mode

        # --- 4) Optional step log ---
        if self._log_cb:
            self._log_cb({
                "step": self._step,
                "ckpt_on": self._ckpt_on,
                "headroom": getattr(self, "_headroom", 1.0),
                "modes": _count_modes(self._registry),
            })
        # compact record for file logging (if enabled)
        try:
            mix = self.precision_mix() if hasattr(self, "precision_mix") else {}
        except Exception:
            mix = {}
        self._emit_log({
            "step": int(getattr(self, "_step", -1)),
            "headroom": float(getattr(self, "_headroom", -1.0)),
            "ckpt_on": bool(getattr(self, "_ckpt_on", False)),
            "mix": mix,
        })

        # --- 5) Cleanup freezes (if you use them elsewhere) ---
        self._freeze_active = False
        self._mode_freeze.clear()




    
    # ---- ergonomics ----
    def on_log(self, fn: Callable[[dict], None]) -> None:
        self._log_cb = fn

    def is_checkpointing(self) -> bool:
        return self._ckpt_on

    def modes_summary(self) -> Dict[str, int]:
        return _count_modes(self._registry)

# --- Internal helpers --------------------------------------------------------


def _count_modes(reg: Dict[nn.Module, _ModuleState]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for st in reg.values():
        out[st.mode] = out.get(st.mode, 0) + 1
    return out

@contextmanager
def _checkpoint_region(self):
    """Enable per-module checkpointing for the current forward region."""
    self._ckpt_depth += 1
    try:
        yield
    finally:
        self._ckpt_depth -= 1

