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
    ckpt_low: float = 0.12
    ckpt_high: float = 0.20
    ckpt_need: int = 2         # consecutive decisions required to flip gate

    # Signal collection cadence
    signals_freq_steps: int = 50

    ckpt_preserve_rng_state: bool = True
    determinism_check: str = "default"
    # Device & optional extras
    device_type: str = "cuda"  # "cuda" | "cpu" | "mps"
    allow_fp8: bool = True
    fp8_backend: str = "te"    # currently only "te" (Transformer Engine), if present

    # Distributed aggregation policy (only if torch.distributed is initialized)
    dist_policy: str = "min_rank"  # "min_rank" | "mean"

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



    # ---- public API ----
    def __init__(self, **kwargs):
        self._freeze_active: bool = False
        self._mode_freeze: dict = {}
        self._ckpt_depth = 0  # >0 while inside the checkpoint region context
        self.cfg = DPCSConfig(**kwargs)
        self.device_type = self.cfg.device_type
        # internal state
        self._step = 0
        self._ckpt_depth = 0   
        self._ckpt_on = False
        self._ckpt_votes = 0
        self._registry: Dict[nn.Module, _ModuleState] = {}
        self._log_cb: Optional[Callable[[dict], None]] = None

        # optional FP8 backend availability flag (we stub now; wire later)
        self._fp8_ok = False
        if self.cfg.allow_fp8 and self.cfg.fp8_backend == "te":
            try:
                import transformer_engine.pytorch as te  # type: ignore
                # Minimal gate: runtime import succeeded; finer gating by arch later.  :contentReference[oaicite:2]{index=2}
                self._fp8_ok = True
                self._te = te
            except Exception:
                self._fp8_ok = False
                self._te = None

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


    def collect_signals(self, loss: torch.Tensor, model: nn.Module) -> None:
        """
        Gather cheap signals:
          - recent GradScaler behavior (overflow hint) via scale changes,
          - per-module running grad-norm (cheap proxy for variance, seeded here),
        Heavy curvature probes will be added in a later step (rate-limited).  :contentReference[oaicite:7]{index=7}
        """
        # Initialize per-module accumulators lazily
        for m in self._registry.keys():
            st = self._registry[m]
            if st.grad_l2_ema is None:
                st.grad_l2_ema = 0.0
            # one cheap sample: first param with grad, if present
            for p in m.parameters(recurse=False):
                if p.grad is not None:
                    g2 = float(p.grad.detach().pow(2).mean().item())
                    st.grad_l2_ema = 0.9 * st.grad_l2_ema + 0.1 * g2
                    break

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
        for m, st in self._registry.items():
            mode = st.mode
            if force_fp32:
                mode = "fp32"
            else:
                # Grad variance proxy decision (EMA threshold)
                if st.grad_l2_ema is not None and st.grad_l2_ema < self.cfg.epsilon_g:
                    mode = "fp16"  # safe to stay at lower precision
                else:
                    mode = "fp16"  # default for now; FP8 wired in later
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

@dataclass
class _ModuleState:
    mode: str = "fp16"
    grad_l2_ema: Optional[float] = None

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

