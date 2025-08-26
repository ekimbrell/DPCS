# dpcs.py — Step 1 skeleton (drop-in, no heavy deps)
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
    # Precision scheduling thresholds
    epsilon_g: float = 1e-3    # grad variance proxy threshold
    kappa: float = 5.0         # loss curvature proxy threshold

    # Activation checkpointing gate (hysteresis)
    # ckpt_low: float = 0.12
    # ckpt_high: float = 0.20
    ckpt_need: int = 2         # consecutive decisions required to flip gate

    # Signal collection cadence
    signals_freq_steps: int = 50

    ckpt_preserve_rng_state: bool = True
    determinism_check: str = "default"
    # Device & optional extras
    device_type: str = "cuda"  # "cuda" | "cpu" | "mps"
    # allow_fp8: bool = True
    fp8_backend: str = "te"    # currently only "te" (Transformer Engine), if present

    # Distributed aggregation policy (only if torch.distributed is initialized)
    dist_policy: str = "min_rank"  # "min_rank" | "mean"
    device_type: str = "cuda"

    # checkpoint band (you likely already have analogous fields)
    ckpt_low: float = 0.15
    ckpt_high: float = 0.30

    # precision scheduler (new)
    enable_precision: bool = True
    epsilon_g: float = 1e-4       # gradient variance threshold (ε_g)
    kappa: float = 1e-1           # curvature proxy threshold (κ)
    cooldown_steps: int = 20      # fp32 cooldown after instability
    ema_beta: float = 0.9         # EMA smoothing for stats
    signals_freq_steps: int = 1   # update frequency

    # FP8 (optional)
    allow_fp8: bool = False
    fp8_backend: str = "te"       # ("te" for NVIDIA Transformer Engine)

class DPCS:
    """
    Drop-in scheduler that:
      - decides per-module precision (fp32/fp16/fp8) for the next step,
      - decides whether to enable activation checkpointing for the next step.

    Usage:
        dpcs = DPCS(...)
        model = dpcs.wrap(model)
        for ...:
            dpcs.start_step()
            fwd_ctx, _ = dpcs.checkpoint_contexts_if_needed()
            with fwd_ctx, torch.autocast(device_type=dpcs.device_type):
                loss = model(x).mean()
            scaler.scale(loss).backward()
            dpcs.collect_signals(loss, model)
            scaler.step(optim); scaler.update()
            dpcs.end_step(optim, scaler)
    """
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

    def _update_stats_and_choose_mode(self, mod):
        st = self._registry[mod]
        params = self._module_params(mod)
        if not params:
            return

        with torch.no_grad():
            gcat = torch.cat([p.grad.detach().float().reshape(-1) for p in params if p.grad is not None], dim=0)
            if gcat.numel() == 0:
                return

            gnorm = torch.linalg.vector_norm(gcat).item() + 1e-12
            gvar  = float(gcat.var(unbiased=False).item()) if gcat.numel() > 1 else 0.0
            prev  = st.gnorm_ema if st.gnorm_ema > 0 else gnorm
            curv_proxy = abs(math.log(gnorm / (prev + 1e-12)))

            st.gnorm_ema = self._ema_update(st.gnorm_ema, gnorm, self._ema_beta)
            st.gvar_ema  = self._ema_update(st.gvar_ema,  gvar,  self._ema_beta)
            st.curv_ema  = self._ema_update(st.curv_ema,  curv_proxy, self._ema_beta)

            # Instability/overflow → force FP32 and start cooldown
            if not self._grads_finite(mod) or st.curv_ema > (5 * self._kappa):
                st.mode = "fp32"
                st.cool = self._cool     # <— THIS drives the test's expectation
                st.overflow_count += 1
                st.last_update_step = self._step
                return

            # Consume cooldown: stay in fp32 while cool > 0
            if st.cool > 0:
                st.cool -= 1
                st.mode = "fp32"
                st.last_update_step = self._step
                return

            # Decide precision (simple ε_g / κ rule)
            if (st.gvar_ema < self._eps_g) or (st.curv_ema > self._kappa):
                st.mode = "fp16" if not getattr(self, "_fp8_ok", False) else "fp8"
            else:
                st.mode = "fp32"

            st.last_update_step = self._step
        
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
        self._ckpt_depth += 1
        try:
            yield
        finally:
            self._ckpt_depth -= 1



    def __init__(self, **kwargs):
        # freeze/override toggles
        self._freeze_active: bool = False
        self._mode_freeze: dict = {}

        # config
        self.cfg = DPCSConfig(**kwargs)
        self.device_type = self.cfg.device_type

        # step / bookkeeping
        self._step = 0
        self._ckpt_depth = 0
        self._ckpt_on = False
        self._ckpt_votes = 0

        # module registry: each wrapped module gets a per-module state (_ModuleState)
        self._registry: Dict[nn.Module, _ModuleState] = {}

        # optional logger callback
        self._log_cb: Optional[Callable[[dict], None]] = None

        # ---------- Checkpoint controller (hysteresis band) ----------
        # Turn checkpointing on when headroom < ckpt_low; keep it on until headroom > ckpt_high.
        self._ckpt_low = float(self.cfg.ckpt_low)
        self._ckpt_high = float(self.cfg.ckpt_high)

        # ---------- Precision scheduler ----------
        # Global enable and thresholds for per-module FP32/FP16(/FP8) decisions
        self._prec_on: bool = bool(self.cfg.enable_precision)
        self._eps_g: float = float(self.cfg.epsilon_g)     # grad variance threshold
        self._kappa: float = float(self.cfg.kappa)         # curvature proxy threshold
        self._cool: int = int(self.cfg.cooldown_steps)     # fp32 cooldown steps
        self._ema_beta: float = float(self.cfg.ema_beta)   # EMA smoothing
        self._signals_freq_steps: int = int(self.cfg.signals_freq_steps)

        # ---------- Optional FP8 backend (Transformer Engine) ----------
        # We gate FP8 by (a) user allowing it, (b) backend selection, (c) importability.
        self._fp8_ok = False
        self._te = None
        if self.cfg.allow_fp8 and self.cfg.fp8_backend == "te":
            try:
                import transformer_engine.pytorch as te  # type: ignore
                # Keep it lazy: only "ok" here; actual FP8 usage still checks device/shape support.
                self._fp8_ok = True
                self._te = te
            except Exception:
                self._fp8_ok = False
                self._te = None

    # (Optional) You can sanity-warn if user enabled precision scheduling without CUDA:
    # if self._prec_on and self.device_type != "cuda":
    #     warnings.warn("DPCS precision scheduling is optimized for CUDA autocast/GradScaler.", RuntimeWarning)


    def wrap(self, model: nn.Module, allow_fp8: Optional[bool] = None) -> nn.Module:
        """Install tiny forward shims on leaf modules to allow local dtype overrides."""
        if allow_fp8 is not None:
            self._fp8_ok = self._fp8_ok and allow_fp8

        for m in model.modules():
            # Only wrap block-level modules:
            should = (
                isinstance(m, nn.TransformerEncoderLayer)  # known good block boundary
                or _is_block_sequential(m)                 # sequential whose children are all leaves
            )
            if not should:
                continue
            if hasattr(m, "_dpcs_orig_forward"):
                continue

            # register and save original forward
            self._registry[m] = self._registry.get(m, _ModuleState(mode="fp16"))  # keep your existing state class
            m._dpcs_orig_forward = m.forward  # type: ignore[attr-defined]
            

            def _wrapped_forward(mod=m):
                def fwd(*args, **kwargs):
                    # choose mode; honor freeze during this step
                    st = self._registry.get(mod)
                    if self._freeze_active and mod in self._mode_freeze:
                        mode = self._mode_freeze[mod]
                    else:
                        mode = st.mode if st else "fp16"

                    def run_with_local_autocast(*a, **k):
                        if getattr(self, "device_type", "cuda") == "cuda":
                            if mode == "fp32":
                                with torch.autocast(device_type="cuda", enabled=False):
                                    return mod._dpcs_orig_forward(*a, **k)
                            elif mode == "fp8" and getattr(self, "_fp8_ok", False):
                                # TODO: swap to TE's fp8_autocast later; fp16 local path for now
                                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                                    return mod._dpcs_orig_forward(*a, **k)
                            else:
                                # fp16 default
                                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                                    return mod._dpcs_orig_forward(*a, **k)
                        else:
                            # CPU/MPS: call through; external autocast governs
                            return mod._dpcs_orig_forward(*a, **k)

                    if self._ckpt_depth > 0 and torch.is_grad_enabled():
                        return checkpoint(
                            run_with_local_autocast, *args,
                            use_reentrant=False,          # explicit per PyTorch guidance
                            preserve_rng_state=True,      # keep dropout determinism unless profiling pure speed
                            determinism_check="none",     # relax metadata equality check during bring-up
                            **kwargs
                        )
                    else:
                        return run_with_local_autocast(*args, **kwargs)
                    self._registry[mod] = _ModuleState(mode="fp16")
                    mod._dpcs_orig_forward = mod.forward
                return fwd

            m.forward = _wrapped_forward()
            assert callable(m.forward), f"DPCS wrap failed on {m.__class__.__name__}"

        return model
    
    
    def start_step(self) -> None:
        self._step += 1
        # (cheap) snapshot headroom once per step (normalized free/total)  :contentReference[oaicite:5]{index=5}
        self._headroom = _mem_headroom(self.device_type)
        self._mode_freeze = {m: st.mode for m, st in self._registry.items()}
        self._freeze_active = True

    def checkpoint_contexts_if_needed(self):
        if self._ckpt_on:
            return self._checkpoint_region(), nullcontext()
        return nullcontext(), nullcontext()


    def collect_signals(self, loss: torch.Tensor, model: torch.nn.Module):
        if not getattr(self, "_prec_on", False):
            return
        
        for mod in self._registry.keys():
            try:
                st = self._registry[mod]
                if st.last_update_step != self._step:
                    self._update_stats_and_choose_mode(mod)
                    # inside your stats update
                if not self._grads_finite(mod) or st.curv_ema > (5 * self._kappa):
                    st.mode = "fp32"
                    st.cool = self._cool
                    st.overflow_count += 1
                    st.last_update_step = self._step
                    return

            except Exception:
                # conservative fallback
                st = self._registry[mod]
                st.mode = "fp32"
                st.cool = max(st.cool, self._cool)
                st.last_update_step = self._step

    def end_step(self, optim: torch.optim.Optimizer, scaler: Optional[torch.amp.GradScaler] = None) -> None:
        """
        Decide next-step precision per module + checkpointing gate using:
          - device headroom (hysteresis gate),
          - recent overflow hint (if GradScaler scale dropped),
          - grad variance proxy (EMA on grad L2; full var/curvature later).
        """
        # 1) Checkpointing gate via hysteresis on headroom
        low, high, need = self.cfg.ckpt_low, self.cfg.ckpt_high, self.cfg.ckpt_need
        vote_on = (self._headroom <= low)
        vote_off = (self._headroom >= high)
        if vote_on:
            self._ckpt_votes = max(0, self._ckpt_votes) + 1
        elif vote_off:
            self._ckpt_votes = min(0, self._ckpt_votes) - 1
        # Flip only after enough consecutive votes
        if self._ckpt_votes >= need:  self._ckpt_on = True;  self._ckpt_votes = 0
        if self._ckpt_votes <= -need: self._ckpt_on = False; self._ckpt_votes = 0

        # 2) Overflow hint from GradScaler (if provided):  :contentReference[oaicite:8]{index=8}
        force_fp32 = False
        if scaler is not None:
            # Heuristic: if scale just dropped, treat as overflow signal
            if hasattr(scaler, "get_scale") and hasattr(self, "_last_scale"):
                try:
                    cur = float(scaler.get_scale())
                    if cur < getattr(self, "_last_scale"):
                        force_fp32 = True
                    self._last_scale = cur
                except Exception:
                    pass
            else:
                try:
                    self._last_scale = float(scaler.get_scale())
                except Exception:
                    self._last_scale = None

        # 3) Per-module next-mode (simple seed policy; refine later)
        # 3) Per-module next-mode (simple seed policy; refine later)
        for m, st in self._registry.items():
            # defensive: older states might be missing fields
            if not hasattr(st, "cool"): st.cool = 0
            if not hasattr(st, "gvar_ema"): st.gvar_ema = 0.0
            if not hasattr(st, "curv_ema"): st.curv_ema = 0.0

            # Start from current mode
            mode = st.mode

            if force_fp32:
                # scaler hinted an overflow this step → pin fp32 and start cooldown
                mode = "fp32"
                st.cool = max(st.cool, self._cool)
            elif st.cool > 0:
                # still cooling down after instability → stay in fp32
                st.cool -= 1
                mode = "fp32"
            else:
                # Decision rule (matches your project spec):
                # lower precision when gradient variance is small (gvar_ema < εg)
                # or curvature proxy is large (curv_ema > κ)
                if (st.gvar_ema < self.cfg.epsilon_g) or (st.curv_ema > self.cfg.kappa):
                    if self.cfg.allow_fp8 and getattr(self, "_fp8_ok", False):
                        mode = "fp8"
                    else:
                        mode = "fp16"
                else:
                    mode = "fp32"

            st.mode = mode


        # 4) Optional log callback
        if self._log_cb:
            self._log_cb({
                "step": self._step,
                "ckpt_on": self._ckpt_on,
                "headroom": getattr(self, "_headroom", 1.0),
                "modes": _count_modes(self._registry),
            })
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

