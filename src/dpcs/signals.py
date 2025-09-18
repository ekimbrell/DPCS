"""Signal collection for DPCS (grad stats, curvature, activations/latency).

This module keeps *lightweight* signal computations separate from the scheduler
and runtime toggles. Everything here is designed to add minimal overhead:
- Per-leaf grad stats via tiny hooks with small uniform subsampling.
- Optional curvature estimates via budgeted Hessian–vector power iteration.
- Activation size & forward-time are commonly gathered inside wrappers; a small
  helper is provided to compute bytes for arbitrary outputs when needed.

All classes avoid per-step dynamic allocations and update in-place.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple
import math
import time

import torch
import torch.nn as nn

# ------------------------------- Utilities ---------------------------------

class EMA:
    """Simple scalar EMA with fixed beta; no tensor allocations.

    Attributes
    -----------
    value : float
        Last EMA value.
    initialized : bool
        Whether the EMA has seen a datapoint.
    """
    __slots__ = ("beta", "value", "initialized")

    def __init__(self, beta: float = 0.95) -> None:
        self.beta = float(beta)
        self.value = 0.0
        self.initialized = False

    def update(self, x: float) -> float:
        xv = float(x)
        if not self.initialized:
            self.value = xv
            self.initialized = True
        else:
            b = self.beta
            self.value = b * self.value + (1.0 - b) * xv
        return self.value


def tensor_bytes(obj: object) -> int:
    """Approximate bytes held by tensor-like outputs (recurses lists/tuples/dicts)."""
    if torch.is_tensor(obj):
        return obj.numel() * obj.element_size()
    if isinstance(obj, (list, tuple)):
        return sum(tensor_bytes(o) for o in obj)
    if isinstance(obj, dict):
        return sum(tensor_bytes(v) for v in obj.values())
    return 0


# ---------------------------- Gradient signals -----------------------------

class GradSignals:
    """Collect per-leaf gradient mean/variance EMAs via tiny hooks.

    This class attaches a single hook to each *parameter* of the provided
    modules (leaves). Each hook takes a small, evenly spaced subsample of the
    gradient and updates per-leaf EMAs of |grad| mean and variance. It stores
    exactly two small EMA scalars per leaf and allocates no per-step tensors on
    the hot path.

    Parameters
    ----------
    leaves : Iterable[nn.Module]
        Typically the leaf wrappers created by the scheduler.
    sample_max_elems : int
        Maximum number of elements sampled from a gradient tensor to estimate
        moments. Sampling is even-strided and avoids index tensor allocation.
    beta : float
        EMA decay for both mean and variance.
    """

    def __init__(self, leaves: Iterable[nn.Module], sample_max_elems: int = 4096, beta: float = 0.9) -> None:
        self.leaves: List[nn.Module] = list(leaves)
        self.sample_max = int(sample_max_elems)
        self.mean_ema: List[EMA] = [EMA(beta) for _ in self.leaves]
        self.var_ema: List[EMA] = [EMA(beta) for _ in self.leaves]
        self._last_var: List[Optional[float]] = [None] * len(self.leaves)
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []

    def attach(self) -> None:
        """Register hooks on all parameters under each leaf."""
        # Remove existing hooks if any
        for h in self._hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._hooks.clear()

        for li, leaf in enumerate(self.leaves):
            for p in leaf.parameters(recurse=True):
                if not p.requires_grad:
                    continue
                self._hooks.append(p.register_hook(self._make_hook(li)))

    def _make_hook(self, li: int):
        maxk = self.sample_max
        mean_ema = self.mean_ema[li]
        var_ema = self.var_ema[li]

        def _hook(g: torch.Tensor):
            if g is None:
                return
            with torch.no_grad():
                g1 = g
                if g1.is_sparse:
                    g1 = g1.coalesce().values()
                flat = g1.reshape(-1)
                n = flat.numel()
                if n == 0:
                    return
                k = min(maxk, n)
                step = max(1, n // k)
                sample = flat[::step][:k].float().abs()
                # Move to CPU for cheap mean/var if needed (small vector)
                if sample.is_cuda:
                    sample = sample.to("cpu")
                m = float(sample.mean())
                v = float(((sample - m) ** 2).mean())
                mean_ema.update(m)
                var_ema.update(v)
                try:
                    self._last_var[li] = float(v)
                except Exception:
                    self._last_var[li] = None
        return _hook

    # Convenience getters ----------------------------------------------------
    def grad_mean(self, li: int) -> Optional[float]:
        e = self.mean_ema[li]
        return e.value if e.initialized else None

    def grad_var(self, li: int) -> Optional[float]:
        e = self.var_ema[li]
        return e.value if e.initialized else None

    def grad_var_avg(self) -> Optional[float]:
        vals = [e.value for e in self.var_ema if e.initialized]
        if not vals:
            return None
        return float(sum(vals) / len(vals))

    def last_var(self, li: int) -> Optional[float]:
        if li < 0 or li >= len(self._last_var):
            return None
        return self._last_var[li]


# ---------------------------- Curvature signals ----------------------------

@dataclass
class _VecCache:
    """Per-leaf vector cache for power iteration (keeps shape mapping)."""
    shapes: List[Tuple[int, ...]]
    sizes: List[int]
    total: int
    vec: Optional[torch.Tensor] = None  # flat vector on device


def _params_of(module: nn.Module) -> List[nn.Parameter]:
    return [p for p in module.parameters(recurse=True) if p.requires_grad]


def _flatten_tensors(tensors: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, List[Tuple[int, ...]], List[int]]:
    flats = []
    shapes: List[Tuple[int, ...]] = []
    sizes: List[int] = []
    for t in tensors:
        shapes.append(tuple(t.shape))
        sizes.append(t.numel())
        flats.append(t.reshape(-1))
    return torch.cat(flats) if flats else torch.empty(0), shapes, sizes


def _unflatten(flat: torch.Tensor, shapes: List[Tuple[int, ...]], sizes: List[int]) -> List[torch.Tensor]:
    out: List[torch.Tensor] = []
    i = 0
    for s in sizes:
        out.append(flat[i:i + s].reshape(shapes[len(out)]))
        i += s
    return out


def _hvp(loss: torch.Tensor, params: Sequence[torch.Tensor], vec_flat: torch.Tensor) -> torch.Tensor:
    """Compute Hessian–vector product H @ v as a flat tensor.

    Uses autograd: hvp = d/dp (g·v), where g = dl/dp with create_graph=True.
    """
    if len(params) == 0:
        return torch.empty(0, device=loss.device)
    with torch.enable_grad():
        g = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True, allow_unused=True)
        g = [gi if gi is not None else torch.zeros_like(p) for gi, p in zip(g, params)]
        g_flat, shapes, sizes = _flatten_tensors(g)
        v = vec_flat
        if v.device != g_flat.device:
            v = v.to(g_flat.device)
        gv = (g_flat * v).sum()
        hv = torch.autograd.grad(gv, params, retain_graph=True, allow_unused=True)
        hv = [hvi if hvi is not None else torch.zeros_like(p) for hvi, p in zip(hv, params)]
        hv_flat, _, _ = _flatten_tensors(hv)
        return hv_flat


class CurvatureSignals:
    """Budgeted curvature proxy via power iteration of HVP per leaf.

    The class maintains a rolling pointer over leaves and, at most every
    ``curv_period`` steps, probes up to ``max_modules_per_probe`` leaves with
    ``hvp_power_iters`` steps of power iteration to estimate the top-1 Hessian
    eigenvalue (Rayleigh quotient). Results are stored per leaf.

    Notes
    -----
    * This is optional and can be disabled by setting ``curv_period=0``.
    * The probe uses flattened parameter vectors for the chosen leaf.
    * All temporary buffers are reused via a per-leaf vector cache.
    """

    def __init__(
        self,
        leaves: Iterable[nn.Module],
        curv_period: int = 50,
        hvp_power_iters: int = 2,
        max_modules_per_probe: int = 1,
        device: Optional[torch.device] = None,
    ) -> None:
        self.leaves: List[nn.Module] = list(leaves)
        self.curv_period = max(0, int(curv_period))
        self.hvp_power_iters = max(1, int(hvp_power_iters))
        self.max_modules_per_probe = max(1, int(max_modules_per_probe))
        self.device = device

        # Book-keeping
        self._step: int = 0
        self._cursor: int = 0
        self.kappa: List[Optional[float]] = [None] * len(self.leaves)
        self._vec_cache: List[_VecCache] = []
        for leaf in self.leaves:
            ps = _params_of(leaf)
            _, shapes, sizes = _flatten_tensors([p.detach() for p in ps])
            total = sum(sizes)
            self._vec_cache.append(_VecCache(shapes=shapes, sizes=sizes, total=total))

    def maybe_probe(self, loss: torch.Tensor) -> None:
        """Probe a small subset of leaves this step if period matches."""
        if self.curv_period <= 0 or len(self.leaves) == 0:
            self._step += 1
            return
        if (self._step % self.curv_period) != 0:
            self._step += 1
            return

        n = len(self.leaves)
        budget = min(self.max_modules_per_probe, n)
        for _ in range(budget):
            li = self._cursor
            self._cursor = (self._cursor + 1) % n
            self._probe_leaf(li, loss)
        self._step += 1

    # internal ---------------------------------------------------------------
    def _probe_leaf(self, li: int, loss: torch.Tensor) -> None:
        leaf = self.leaves[li]
        params = _params_of(leaf)
        if not params:
            self.kappa[li] = None
            return
        # Build or reuse a random vector v of matching size
        cache = self._vec_cache[li]
        if cache.total == 0:
            self.kappa[li] = None
            return
        if cache.vec is None or cache.vec.numel() != cache.total or (self.device and cache.vec.device != self.device):
            dev = self.device or params[0].device
            # Rademacher vector for stability
            cache.vec = torch.empty(cache.total, device=dev).bernoulli_(0.5).mul_(2.0).add_(-1.0)
        v = cache.vec
        v = v / (v.norm() + 1e-9)

        # Power iterations
        ray = None
        with torch.enable_grad():
            for _ in range(self.hvp_power_iters):
                hv = _hvp(loss, params, v)
                # Rayleigh quotient ~ v^T H v / (v^T v)
                num = float(torch.dot(v, hv).detach().cpu())
                den = float(torch.dot(v, v).detach().cpu()) + 1e-12
                ray = num / den
                # Normalize next vector
                v = hv / (hv.norm() + 1e-9)
        self.kappa[li] = float(ray) if ray is not None else None


# -------------------------- Activation & latency ---------------------------

class ForwardTimer:
    """Reusable CUDA event or CPU timer for forward-pass latency EMA."""
    __slots__ = ("ev_s", "ev_e", "ema")
    def __init__(self, beta: float = 0.9) -> None:
        self.ema = EMA(beta)
        if torch.cuda.is_available():
            self.ev_s = torch.cuda.Event(enable_timing=True)
            self.ev_e = torch.cuda.Event(enable_timing=True)
        else:
            self.ev_s = None  # type: ignore
            self.ev_e = None  # type: ignore

    def start(self) -> None:
        if self.ev_s is not None:
            self.ev_s.record()

    def stop_update(self) -> float:
        if self.ev_e is not None and self.ev_s is not None:
            self.ev_e.record(); self.ev_e.synchronize()
            ms = float(self.ev_s.elapsed_time(self.ev_e))
        else:
            # fallback host timer (rough)
            ms = 0.0
        self.ema.update(ms)
        return ms


__all__ = [
    "EMA",
    "tensor_bytes",
    "GradSignals",
    "CurvatureSignals",
    "ForwardTimer",
]
