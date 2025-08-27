"""
DPCS Distributed Utilities (scaffold)

This module provides light-weight helpers to:
  • initialize torch.distributed
  • wrap a model with DDP or FSDP
  • (optional) register DDP communication hooks for BF16/FP16 gradient compression
  • gather tiny per-rank stats for global threshold tuning (epsilon_g, kappa)

Notes
-----
- Designed to be *optional*: if torch.distributed is unavailable or uninitialized,
  functions fallback safely and return the original model.
- Keep this file independent from core dpcs.py to avoid circular imports.

Example
-------
>>> from dpcs.distrib import init_distributed, wrap_ddp, ddp_all_gather_list
>>> init_distributed()  # reads RANK / WORLD_SIZE / LOCAL_RANK
>>> model = wrap_ddp(model, comm_hook="bf16")
>>> # training loop ...

This is a scaffold: extend as your project evolves.
"""
from __future__ import annotations

import os
import math
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

# ---- Optional distributed imports -------------------------------------------------
try:
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    _HAS_DIST = True
except Exception:  # pragma: no cover - import-time environment dependent
    dist = None  # type: ignore
    DDP = None  # type: ignore
    _HAS_DIST = False

# FSDP is optional; keep import guarded
try:  # pragma: no cover
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import ShardingStrategy
    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
    _HAS_FSDP = True
except Exception:  # pragma: no cover - FSDP not available
    FSDP = None  # type: ignore
    ShardingStrategy = None  # type: ignore
    size_based_auto_wrap_policy = None  # type: ignore
    _HAS_FSDP = False

# DDP comm hooks (bf16/fp16 compression) are available in PyTorch 1.10+
try:  # pragma: no cover
    from torch.distributed.algorithms.ddp_comm_hooks import default as ddp_default_hooks
    _HAS_COMM_HOOKS = True
except Exception:
    ddp_default_hooks = None  # type: ignore
    _HAS_COMM_HOOKS = False

__all__ = [
    "init_distributed",
    "is_distributed",
    "get_rank",
    "get_world_size",
    "wrap_ddp",
    "wrap_fsdp",
    "register_ddp_comm_hook",
    "ddp_all_gather_list",
    "DistributedSignals",
]

# -----------------------------------------------------------------------------------
# Process group helpers
# -----------------------------------------------------------------------------------

def is_distributed() -> bool:
    return _HAS_DIST and dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    if is_distributed():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    if is_distributed():
        return dist.get_world_size()
    return 1


def init_distributed(
    backend: Optional[str] = None,
    init_method: Optional[str] = None,
    timeout: Optional[float] = None,
    set_device_from_local_rank: bool = True,
    seed: Optional[int] = None,
) -> None:
    """Initialize torch.distributed with common environment variables.

    Environment variables used when arguments are None:
      - RANK, WORLD_SIZE, LOCAL_RANK
      - MASTER_ADDR, MASTER_PORT (when using init_method env://)

    If ``set_device_from_local_rank`` is True and CUDA is available, the CUDA
    current device is set to LOCAL_RANK.
    """
    if not _HAS_DIST or not dist.is_available():
        warnings.warn("torch.distributed not available; running single-process.")
        return

    if dist.is_initialized():
        return

    backend = backend or ("nccl" if torch.cuda.is_available() else "gloo")
    init_method = init_method or "env://"

    # device placement
    if set_device_from_local_rank and torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)

    # timeout is optional (seconds)
    pg_timeout = torch.distributed.constants.default_pg_timeout if timeout is None else torch.timedelta(seconds=timeout)  # type: ignore[attr-defined]

    dist.init_process_group(backend=backend, init_method=init_method, timeout=pg_timeout)

    # rank-aware RNG seeding (optional, reproducible across runs)
    if seed is not None:
        rank = dist.get_rank()
        torch.manual_seed(seed + rank)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + rank)


# -----------------------------------------------------------------------------------
# Wrappers
# -----------------------------------------------------------------------------------

def wrap_ddp(
    model: nn.Module,
    device_ids: Optional[Sequence[int]] = None,
    output_device: Optional[int] = None,
    broadcast_buffers: bool = False,
    static_graph: bool = False,
    bucket_cap_mb: int = 25,
    gradient_as_bucket_view: bool = True,
    comm_hook: Optional[str] = None,  # "bf16" | "fp16" | None
    process_group: Any = None,
) -> nn.Module:
    """Wrap a model in DistributedDataParallel and (optionally) register a comm hook.

    ``comm_hook`` can be one of {"bf16", "fp16", None}. When set, it registers the
    corresponding compression hook on the DDP model.
    """
    if not is_distributed():
        return model

    ddp_model = DDP(
        model,
        device_ids=device_ids,
        output_device=output_device,
        broadcast_buffers=broadcast_buffers,
        static_graph=static_graph,
        bucket_cap_mb=bucket_cap_mb,
        gradient_as_bucket_view=gradient_as_bucket_view,
        process_group=process_group,
    )

    if comm_hook is not None:
        register_ddp_comm_hook(ddp_model, comm_hook, process_group)
    return ddp_model


def register_ddp_comm_hook(ddp_model: nn.Module, hook: str, process_group: Any = None) -> None:
    """Register built-in BF16/FP16 compression hooks on a DDP model.

    - bf16: casts gradients to bfloat16 during communication and restores dtype after reduce
    - fp16: casts gradients to float16 during communication and restores dtype after reduce

    If hooks are unavailable (old PyTorch), a warning is emitted.
    """
    if not _HAS_COMM_HOOKS:
        warnings.warn("DDP comm hooks not available in this PyTorch build; skipping.")
        return
    hook = hook.lower()
    if hook == "bf16":
        ddp_model.register_comm_hook(process_group, ddp_default_hooks.bf16_compress_hook)  # type: ignore[attr-defined]
    elif hook == "fp16":
        ddp_model.register_comm_hook(process_group, ddp_default_hooks.fp16_compress_hook)  # type: ignore[attr-defined]
    else:
        raise ValueError(f"Unknown comm hook: {hook}. Use 'bf16' | 'fp16' | None.")


# ----- FSDP -----------------------------------------------------------------------

def wrap_fsdp(
    model: nn.Module,
    param_size_threshold: int = 1_000_000,
    sharding: Optional["ShardingStrategy"] = None,
    cpu_offload: bool = False,
    mixed_precision_dtype: Optional[torch.dtype] = None,
) -> nn.Module:
    """Wrap a model with FSDP (if available).

    Parameters
    ----------
    param_size_threshold: int
        Auto-wrap modules whose parameter count >= threshold.
    sharding: ShardingStrategy | None
        E.g., ShardingStrategy.FULL_SHARD. If None, PyTorch default is used.
    cpu_offload: bool
        If True, enables CPU offload policy for params.
    mixed_precision_dtype: torch.dtype | None
        If set (e.g., torch.bfloat16), uses FSDP mixed precision for params, grads, and buffers.
    """
    if not is_distributed() or not _HAS_FSDP:
        return model

    # Auto-wrap policy by parameter size
    def _auto_wrap_policy(module: nn.Module, recurse: bool, unwrapped_params: int) -> bool:
        if size_based_auto_wrap_policy is None:
            return False
        return size_based_auto_wrap_policy(
            {"min_num_params": param_size_threshold}
        )(module, recurse, unwrapped_params)

    mp_policy = None
    if mixed_precision_dtype is not None:
        from torch.distributed.fsdp import MixedPrecision  # local import to avoid hard dep
        mp_policy = MixedPrecision(param_dtype=mixed_precision_dtype,
                                   reduce_dtype=mixed_precision_dtype,
                                   buffer_dtype=mixed_precision_dtype)

    fsdp_model = FSDP(
        model,
        sharding_strategy=sharding,
        auto_wrap_policy=_auto_wrap_policy,
        cpu_offload=cpu_offload,
        mixed_precision=mp_policy,
    )
    return fsdp_model


# -----------------------------------------------------------------------------------
# Tiny cross-rank stats sync (for global epsilon_g/kappa tuning)
# -----------------------------------------------------------------------------------

@dataclass
class DistributedSignals:
    """Minimal helper to exchange small per-rank scalars periodically.

    Usage:
    -------
    ds = DistributedSignals(period=10)
    for step in ...:
        # collect local samples (e.g., module-wise variance/curvature or pooled summaries)
        ds.maybe_sync(step, {"gvar": float_val1, "curv": float_val2})
        if ds.last_global is not None:
            eps, kap = ds.compute_thresholds(percentile_g=0.5, percentile_c=0.5,
                                             min_eps=1e-12, min_kappa=1e-12)
            # apply to your DPCS instance: dpcs.cfg.epsilon_g = eps; dpcs.cfg.kappa = kap
    """
    period: int = 10
    group: Any = None

    # memory of last global pool
    last_global: Optional[List[Dict[str, float]]] = None

    def maybe_sync(self, step: int, local_summary: Dict[str, float]) -> Optional[List[Dict[str, float]]]:
        if not is_distributed() or (self.period <= 0):
            self.last_global = [local_summary]
            return self.last_global
        if (step % self.period) != 0:
            return None
        objs = [local_summary]
        out_list: List[Dict[str, float]] = [None] * get_world_size()  # type: ignore
        dist.all_gather_object(out_list, objs[0], group=self.group)  # blocking, tiny payloads
        self.last_global = out_list
        return out_list

    def compute_thresholds(
        self,
        percentile_g: float = 0.5,
        percentile_c: float = 0.5,
        min_eps: float = 1e-12,
        min_kappa: float = 1e-12,
    ) -> Tuple[float, float]:
        """Compute robust thresholds from last_global summaries.

        Expects each item to contain keys {"gvar", "curv"}. Returns (epsilon_g, kappa).
        """
        assert self.last_global is not None and len(self.last_global) > 0
        gvars = torch.tensor([d.get("gvar", 0.0) for d in self.last_global], dtype=torch.float64)
        curvs = torch.tensor([d.get("curv", 0.0) for d in self.last_global], dtype=torch.float64)
        eps = float(torch.quantile(gvars, torch.tensor(percentile_g, dtype=torch.float64)).item()) if gvars.numel() > 0 else min_eps
        kap = float(torch.quantile(curvs, torch.tensor(percentile_c, dtype=torch.float64)).item()) if curvs.numel() > 0 else min_kappa
        return max(eps, min_eps), max(kap, min_kappa)


# -----------------------------------------------------------------------------------
# Utility: gather Python lists/dicts (small payloads)
# -----------------------------------------------------------------------------------

def ddp_all_gather_list(obj: Any, group: Any = None) -> List[Any]:
    """All-gather a picklable Python object across ranks and return a list.

    Intended for *small* payloads only (scalars, tiny dicts). Uses
    :func:`torch.distributed.all_gather_object`, which serializes via pickle and is
    blocking. Avoid GPU tensors here; use tensor collectives for large/typed data.
    """
    if not is_distributed():
        return [obj]
    out_list: List[Any] = [None] * get_world_size()  # type: ignore
    dist.all_gather_object(out_list, obj, group=group)
    return out_list
