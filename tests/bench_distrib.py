#!/usr/bin/env python
"""
bench_distrib.py — Distributed benchmark for DPCS with DDP/FSDP

Launch with torchrun (recommended):
  torchrun --nproc_per_node=4 tests/bench_distrib.py \
    --parallel ddp --model transformer --device cuda \
    --steps 50 --batch 8 --seq 1024 --d-model 512 --nhead 8 --ff 2048 --layers 6 \
    --precision on --ckpt on --topk-frac 0.3 --comm bf16 \
    --sdpa math --optimizer adamw --lr 2e-4 --jsonl runs/ddp_transformer.jsonl

Or FSDP example (bf16 mixed precision inside FSDP):
  torchrun --nproc_per_node=4 tests/bench_distrib.py \
    --parallel fsdp --model transformer --device cuda \
    --steps 50 --batch 8 --seq 1024 --d-model 512 --nhead 8 --ff 2048 --layers 6 \
    --precision on --ckpt on --topk-frac 0.4 --sdpa math \
    --fsdp-mp bf16 --optimizer adamw --lr 2e-4 --jsonl runs/fsdp_transformer.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

# project imports
from dpcs import DPCS
from dpcs.distrib import (
    init_distributed,
    is_distributed,
    get_rank,
    get_world_size,
    wrap_ddp,
    wrap_fsdp,
    ddp_all_gather_list,
)

# ---------------------------- Models -----------------------------------------

class DeepMLP(nn.Module):
    def __init__(self, width=4096, depth=8, out_dim=1000):
        super().__init__()
        layers = [nn.Linear(width, width)]
        for _ in range(depth - 1):
            layers += [nn.GELU(), nn.Linear(width, width)]
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(width, out_dim)

    def forward(self, x):
        x = self.backbone(x)
        return self.head(x)


class TinyTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, dim_ff=2048, nlayers=6, n_classes=1000, dropout=0.0, norm_first=True):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
                                               dropout=dropout, batch_first=True, norm_first=norm_first)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x):  # x: (B, S, D)
        h = self.encoder(x)
        h = h.mean(dim=1)
        return self.head(h)


# ------------------------ SDPA backend control -------------------------------

@contextmanager
def sdpa_kernel_ctx(kind: str):
    """Version-portable SDPA backend selection.
    kind in {"math", "flash", "mem_efficient", "auto"}.
    """
    kind = (kind or "auto").lower()
    # Prefer the new API (PyTorch 2.3+)
    try:
        from torch.nn.attention import sdpa_kernel
        if kind == "math":
            with sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                yield
        elif kind in ("flash", "flash_attention"):
            with sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
                yield
        elif kind in ("mem_efficient", "efficient"):
            with sdpa_kernel(torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION):
                yield
        else:  # auto
            with sdpa_kernel(torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
                             torch.nn.attention.SDPBackend.MATH,
                             torch.nn.attention.SDPBackend.FLASH_ATTENTION):
                yield
        return
    except Exception:
        pass
    # Fallback: no-op context
    yield


# ------------------------ Bench utilities ------------------------------------

@dataclass
class RunConfig:
    model_kind: str
    device: str
    steps: int
    batch: int
    # transformer
    seq: int = 1024
    d_model: int = 512
    nhead: int = 8
    ff: int = 2048
    layers: int = 6
    # mlp
    width: int = 4096
    depth: int = 8
    classes: int = 1000

    optimizer: str = "adamw"  # or sgd
    lr: float = 2e-4
    weight_decay: float = 0.01
    momentum: float = 0.9

    precision: str = "on"      # on/off
    ckpt: str = "on"           # on/off
    topk_frac: float = 0.3
    min_ckpt_bytes: int = 16 << 20

    sdpa: str = "auto"         # auto/math/flash/mem_efficient
    fp8: str = "off"           # off/te

    # distrib
    parallel: str = "ddp"      # ddp/fsdp/none
    comm: str = "bf16"         # bf16/fp16/off
    fsdp_mp: str = "none"      # none/bf16/fp16

    jsonl: Optional[str] = None


def make_model(cfg: RunConfig) -> nn.Module:
    if cfg.model_kind == "transformer":
        return TinyTransformer(d_model=cfg.d_model, nhead=cfg.nhead, dim_ff=cfg.ff,
                               nlayers=cfg.layers, n_classes=cfg.classes)
    elif cfg.model_kind == "mlp":
        return DeepMLP(width=cfg.width, depth=cfg.depth, out_dim=cfg.classes)
    else:
        raise ValueError("Unknown model_kind")


def make_inputs(cfg: RunConfig, device: str):
    if cfg.model_kind == "transformer":
        x = torch.randn(cfg.batch, cfg.seq, cfg.d_model, device=device)
        y = torch.randint(0, cfg.classes, (cfg.batch,), device=device)
    else:  # mlp
        x = torch.randn(cfg.batch, cfg.width, device=device)
        y = torch.randint(0, cfg.classes, (cfg.batch,), device=device)
    return x, y


def make_optimizer(cfg: RunConfig, model: nn.Module):
    if cfg.optimizer == "adamw":
        return optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "sgd":
        return optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum)
    else:
        raise ValueError("Unknown optimizer")


def build_run_id(cfg: RunConfig) -> str:
    p = "p1" if cfg.precision == "on" else "p0"
    c = "c1" if cfg.ckpt == "on" else "c0"
    par = cfg.parallel[:3]
    base = f"{cfg.model_kind[:3]}-{par}-{c}-{p}-{cfg.optimizer}"
    return base


def train_once(cfg: RunConfig) -> Dict:
    rank = get_rank()
    world = get_world_size()
    device = cfg.device

    # DPCS scheduler
    dpcs = DPCS(
        device_type=device,
        enable_precision=(cfg.precision == "on"),
        allow_fp8=(cfg.fp8 == "te"),
        wrap_types=(nn.Linear, nn.TransformerEncoderLayer),
        ckpt_enable_topk=True,
        ckpt_topk_frac=cfg.topk_frac,
        min_activation_bytes_to_ckpt=cfg.min_ckpt_bytes,
    )

    # Model
    model = make_model(cfg).to(device)
    model = dpcs.wrap(model)

    # Distributed wrap
    if cfg.parallel == "ddp":
        model = wrap_ddp(model)
        if cfg.comm in ("bf16", "fp16"):
            # register via wrapper again (safe if already wrapped)
            from dpcs.distrib import register_ddp_comm_hook
            register_ddp_comm_hook(model, cfg.comm)
    elif cfg.parallel == "fsdp":
        mp_dtype = None
        if cfg.fsdp_mp == "bf16":
            mp_dtype = torch.bfloat16
        elif cfg.fsdp_mp == "fp16":
            mp_dtype = torch.float16
        model = wrap_fsdp(model, param_size_threshold=1_000_000, mixed_precision_dtype=mp_dtype)

    # Criterion & Optimizer
    crit = nn.CrossEntropyLoss().to(device)
    opt = make_optimizer(cfg, model)

    # AMP scaler only for CUDA + precision enabled
    use_amp = (device == "cuda" and cfg.precision == "on")
    scaler = torch.amp.GradScaler(device) if use_amp else None

    # Timing / memory
    if device == "cuda":
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()

    t0 = time.time()
    total_samples = 0

    # Training loop
    for step in range(1, cfg.steps + 1):
        dpcs.start_step()
        # checkpoint policy toggle
        dpcs._ckpt_on = (cfg.ckpt == "on")

        x, y = make_inputs(cfg, device)

        model.zero_grad(set_to_none=True)
        # SDPA backend context for Transformers
        sdpa_ctx = sdpa_kernel_ctx(cfg.sdpa) if cfg.model_kind == "transformer" else nullcontext()
        with sdpa_ctx:
            if use_amp:
                with torch.autocast(device_type=device, dtype=(torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)):
                    logits = model(x)
                    loss = crit(logits, y)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward(); opt.step()
            else:
                logits = model(x)
                loss = crit(logits, y)
                loss.backward(); opt.step()

        # stats collection (after backward grads exist)
        dpcs.collect_signals(loss, model)
        dpcs.end_step(opt, scaler)

        total_samples += x.shape[0]

    if device == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()

    # Local metrics
    elapsed = (t1 - t0)
    avg_ms = 1000.0 * elapsed / cfg.steps
    samp_s = total_samples / elapsed
    tok_s = (total_samples * (cfg.seq if cfg.model_kind == "transformer" else 0)) / elapsed
    loss_val = float(loss.detach().item())
    cuda_peak = torch.cuda.max_memory_allocated() if device == "cuda" else 0

    result = {
        "run_id": build_run_id(cfg),
        "rank": rank,
        "world": world,
        "device": device,
        "model_kind": cfg.model_kind,
        "parallel": cfg.parallel,
        "ckpt": (cfg.ckpt == "on"),
        "prec": (cfg.precision == "on"),
        "fp8": (cfg.fp8 != "off"),
        "avg_ms": avg_ms,
        "samp_s": samp_s,
        "tok_s": tok_s,
        "loss": loss_val,
        "cuda_peak": cuda_peak,
    }
    return result


def aggregate_and_print(cfg: RunConfig, local_result: Dict) -> None:
    # Gather results to rank0
    results = ddp_all_gather_list(local_result)
    # Compute global view
    max_peak = max(r["cuda_peak"] for r in results)
    total_samples_s = sum(r["samp_s"] for r in results)
    total_tok_s = sum(r["tok_s"] for r in results)
    loss_avg = sum(r["loss"] for r in results) / len(results)
    avg_ms_rank0 = results[0]["avg_ms"]  # use rank0's avg_ms as reference

    # Pretty print on rank0 only
    if get_rank() == 0:
        hdr = (
            "run_id         device model_kind  parallel ckpt prec fp8 avg_ms   samp/s     tok/s      loss       cuda_peak   "
        )
        print(hdr)
        print("-" * len(hdr))
        print(
            f"{results[0]['run_id']:<14} {cfg.device:<6} {cfg.model_kind:<12} {cfg.parallel:<8} "
            f"{'Y' if cfg.ckpt=='on' else 'N':<3} {'Y' if cfg.precision=='on' else 'N':<3} "
            f"{'Y' if cfg.fp8!='off' else 'N':<3} "
            f"{avg_ms_rank0:7.2f} {total_samples_s:9.2f} {total_tok_s:10.2f} "
            f"{loss_avg:9.4f}  {format_bytes(max_peak):>10}"
        )

        if cfg.jsonl:
            os.makedirs(os.path.dirname(cfg.jsonl) or ".", exist_ok=True)
            with open(cfg.jsonl, "a", buffering=1) as f:
                f.write(json.dumps({
                    "cfg": asdict(cfg),
                    "results": results,
                    "aggregate": {
                        "samp_s": total_samples_s,
                        "tok_s": total_tok_s,
                        "cuda_peak_max": max_peak,
                        "loss_avg": loss_avg,
                        "avg_ms_rank0": avg_ms_rank0,
                    },
                }) + "\n")


def format_bytes(n: int) -> str:
    if n <= 0:
        return "0 B"
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    i = int(math.floor(math.log(n, 1024)))
    i = min(i, len(units) - 1)
    val = n / (1024 ** i)
    return f"{val:.2f} {units[i]}"


# ------------------------ CLI -------------------------------------------------

def parse_args() -> RunConfig:
    p = argparse.ArgumentParser(description="Distributed DPCS benchmark")
    p.add_argument("--parallel", default="ddp", choices=["none", "ddp", "fsdp"]) 
    p.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--model", dest="model_kind", default="transformer", choices=["transformer", "mlp"]) 
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--batch", type=int, default=8)
    # transformer
    p.add_argument("--seq", type=int, default=1024)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--ff", type=int, default=2048)
    p.add_argument("--layers", type=int, default=6)
    # mlp
    p.add_argument("--width", type=int, default=4096)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--classes", type=int, default=1000)

    p.add_argument("--optimizer", default="adamw", choices=["adamw", "sgd"]) 
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--momentum", type=float, default=0.9)

    p.add_argument("--precision", default="on", choices=["on", "off"]) 
    p.add_argument("--ckpt", default="on", choices=["on", "off"]) 
    p.add_argument("--topk-frac", type=float, default=0.3)
    p.add_argument("--min-ckpt-bytes", type=str, default="16MiB")

    p.add_argument("--sdpa", default="auto", choices=["auto", "math", "flash", "mem_efficient"]) 
    p.add_argument("--fp8", default="off", choices=["off", "te"]) 

    p.add_argument("--comm", default="bf16", choices=["bf16", "fp16", "off"]) 
    p.add_argument("--fsdp-mp", default="none", choices=["none", "bf16", "fp16"]) 

    p.add_argument("--jsonl", default=None)

    args = p.parse_args()

    min_bytes = parse_bytes(args.min_ckpt_bytes)

    return RunConfig(
        model_kind=args.model_kind,
        device=args.device,
        steps=args.steps,
        batch=args.batch,
        seq=args.seq,
        d_model=args.d_model,
        nhead=args.nhead,
        ff=args.ff,
        layers=args.layers,
        width=args.width,
        depth=args.depth,
        classes=args.classes,
        optimizer=args.optimizer,
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        precision=args.precision,
        ckpt=args.ckpt,
        topk_frac=args.topk_frac,
        min_ckpt_bytes=min_bytes,
        sdpa=args.sdpa,
        fp8=args.fp8,
        parallel=args.parallel,
        comm=args.comm,
        fsdp_mp=args.fsdp_mp,
        jsonl=args.jsonl,
    )


def parse_bytes(s: str) -> int:
    s = s.strip().lower()
    if s.endswith("mib"):  # binary
        return int(float(s[:-3]) * (1<<20))
    if s.endswith("mb"):   # decimal
        return int(float(s[:-2]) * 1_000_000)
    if s.endswith("gib"):
        return int(float(s[:-3]) * (1<<30))
    if s.endswith("gb"):
        return int(float(s[:-2]) * 1_000_000_000)
    if s.endswith("kib"):
        return int(float(s[:-3]) * (1<<10))
    if s.endswith("kb"):
        return int(float(s[:-2]) * 1_000)
    return int(s)


def main():
    cfg = parse_args()

    # Initialize distributed if available
    init_distributed()

    # Device check
    if cfg.device == "cuda" and not torch.cuda.is_available():
        if get_rank() == 0:
            print("[warn] CUDA requested but unavailable; falling back to CPU.")
        cfg.device = "cpu"

    # Run local training
    local_res = train_once(cfg)

    # Aggregate and print on rank0
    aggregate_and_print(cfg, local_res)


if __name__ == "__main__":
    main()
