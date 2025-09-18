#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
import sys
import time
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from dpcs import DPCS

# ------------------------------- Utils ---------------------------------------

def set_seed(seed=0):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def human_bytes(n: int) -> str:
    if n is None:
        return "0 B"
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    s = float(n)
    for u in units:
        if abs(s) < 1024.0 or u == units[-1]:
            return f"{s:.2f} {u}" if u != "B" else f"{int(s)} B"
        s /= 1024.0


# ------------------------------- Models --------------------------------------

class DeepMLP(nn.Module):
    """Activation-heavy MLP to magnify checkpointing effects."""
    def __init__(self, width=4096, depth=8, out_dim=1000):
        super().__init__()
        layers = [nn.Linear(width, width)]
        for _ in range(depth - 1):
            layers += [nn.GELU(), nn.Linear(width, width)]
        layers += [nn.GELU(), nn.Linear(width, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TinyTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, dim_ff=2048, nlayers=6, n_classes=1000, dropout=0.0):
        super().__init__()
        self.proj_in = nn.Linear(d_model, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
                                               dropout=dropout, activation="gelu", batch_first=True,
                                               norm_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=nlayers, enable_nested_tensor=False)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x):
        # x: (B, S, D)
        h = self.proj_in(x)
        h = self.enc(h)
        # pool last token (CLS-like)
        h = h[:, -1, :]
        return self.head(h)


# -------------------------- SDPA selection (portable) ------------------------

@contextmanager
def sdpa_context(kind: str):
    """Portable SDPA backend chooser. kind in {"math","efficient","flash"}."""
    # PyTorch 2.1+: torch.nn.attention.sdpa_kernel
    try:
        from torch.nn.attention import sdpa_kernel, SDPBackend  # type: ignore
        mapping = {
            "math": [SDPBackend.MATH],
            "efficient": [SDPBackend.EFFICIENT_ATTENTION],
            "flash": [SDPBackend.FLASH_ATTENTION],
        }
        backends = mapping.get(kind, [SDPBackend.MATH])
        with sdpa_kernel(backends if len(backends) > 1 else backends[0]):
            yield
        return
    except Exception:
        pass
    # Older fallback: torch.backends.cuda.sdp_kernel (kwargs variant)
    try:
        from torch.backends.cuda import sdp_kernel  # type: ignore
        if kind == "math":
            ctx = sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)
        elif kind == "efficient":
            ctx = sdp_kernel(enable_flash=False, enable_mem_efficient=True, enable_math=False)
        else:  # flash
            ctx = sdp_kernel(enable_flash=True, enable_mem_efficient=False, enable_math=False)
        with ctx:
            yield
        return
    except Exception:
        pass
    # No-op
    yield


# ------------------------------- Config --------------------------------------

@dataclass
class BenchConfig:
    model_kind: str
    device: str
    steps: int
    batch: int
    # MLP
    width: int = 4096
    depth: int = 8
    classes: int = 1000
    # Transformer
    seq: int = 1024
    d_model: int = 512
    nhead: int = 8
    ff: int = 2048
    layers: int = 6
    # training
    optimizer: str = "sgd"  # or adamw
    lr: float = 3e-4
    weight_decay: float = 0.0
    momentum: float = 0.0
    # policies
    sdpa: str = "math"  # math|efficient|flash
    enable_precision: bool = False
    enable_ckpt: bool = False
    allow_fp8: bool = False


# ------------------------------- Runner --------------------------------------

def _build_model(cfg: BenchConfig) -> Tuple[nn.Module, nn.Module, int, int]:
    if cfg.model_kind == "mlp":
        model = DeepMLP(cfg.width, cfg.depth, cfg.classes)
        crit = nn.CrossEntropyLoss()
        sample_shape = (cfg.batch, cfg.width)
        tokens_per_sample = 0
    elif cfg.model_kind == "transformer":
        model = TinyTransformer(cfg.d_model, cfg.nhead, cfg.ff, cfg.layers, cfg.classes)
        crit = nn.CrossEntropyLoss()
        sample_shape = (cfg.batch, cfg.seq, cfg.d_model)
        tokens_per_sample = cfg.seq
    else:
        raise ValueError("unknown model_kind")
    return model, crit, sample_shape, tokens_per_sample


def _wrap_model_with_dpcs(model: nn.Module, device: str, cfg: BenchConfig) -> DPCS:
    wrap_types = (nn.Linear,) if cfg.model_kind == "mlp" else (nn.Linear, nn.TransformerEncoderLayer)
    dpcs = DPCS(device_type=device,
                enable_precision=cfg.enable_precision,
                wrap_types=wrap_types,
                allow_fp8=cfg.allow_fp8)
    model = dpcs.wrap(model)
    if not cfg.enable_precision:
        dpcs.force_fp32()
    # set ckpt policy
    dpcs._ckpt_on = bool(cfg.enable_ckpt)
    return dpcs


def _make_optimizer(params, cfg: BenchConfig):
    if cfg.optimizer == "adamw":
        try:
            return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay, fused=True)
        except Exception:
            return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(params, lr=cfg.lr, weight_decay=cfg.weight_decay, momentum=cfg.momentum)
    raise ValueError("unknown optimizer")


def _amp_dtype(device: str):
    if device == "cuda" and torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def _train_loop(cfg: BenchConfig, jsonl_path: Optional[str]) -> Dict[str, Any]:
    device = cfg.device
    model, crit, sample_shape, tokens_per_sample = _build_model(cfg)
    model = model.to(device)

    # DPCS wrapping
    dpcs = _wrap_model_with_dpcs(model, device, cfg)

    # Optimizer
    opt = _make_optimizer(model.parameters(), cfg)

    # AMP settings
    use_amp = (device == "cuda" and cfg.enable_precision)
    amp_dtype = _amp_dtype(device)
    scaler = torch.amp.GradScaler(device if device == "cuda" else None,
                                  enabled=(use_amp and amp_dtype == torch.float16))

    # Data
    x = torch.randn(*sample_shape, device=device)
    y = torch.randint(0, cfg.classes, (cfg.batch,), device=device)

    # Memory metrics
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    # Train
    steps = cfg.steps
    start = time.perf_counter()
    with sdpa_context(cfg.sdpa):
        for _ in range(steps):
            dpcs.start_step()
            model.zero_grad(set_to_none=True)
            if use_amp:
                with torch.amp.autocast(device_type=device, dtype=amp_dtype, enabled=True):
                    out = model(x)
                    loss = crit(out, y)
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    opt.step()
            else:
                out = model(x)
                loss = crit(out, y)
                loss.backward()
                opt.step()
            dpcs.end_step(opt, scaler if scaler.is_enabled() else None)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    # Metrics
    avg_ms = (elapsed / steps) * 1000.0
    samp_s = cfg.batch * steps / elapsed
    tok_s = (cfg.batch * tokens_per_sample * steps / elapsed) if tokens_per_sample > 0 else 0.0
    cuda_peak_alloc = torch.cuda.max_memory_allocated(device) if device == "cuda" else 0
    cuda_peak_resv = torch.cuda.max_memory_reserved(device) if device == "cuda" else 0

    # One eval loss (FP32) for sanity
    model.eval()
    with torch.no_grad():
        if device == "cuda":
            torch.cuda.synchronize()
        out = model(x.float()) if use_amp else model(x)
        eval_loss = float(crit(out, y).item())
        if device == "cuda":
            torch.cuda.synchronize()
    model.train()

    # Emit row
    row = {
        "run_id": None,  # set by caller
        "device": device,
        "model_kind": cfg.model_kind,
        "ckpt": "Y" if cfg.enable_ckpt else "N",
        "prec": "Y" if cfg.enable_precision else "N",
        "fp8": "Y" if cfg.allow_fp8 else "N",
        "avg_ms": round(avg_ms, 2),
        "samp_s": round(samp_s, 2),
        "tok_s": round(tok_s, 2),
        "loss": round(float(loss.item()), 4),
        "eval_loss": round(eval_loss, 4),
        "cuda_peak_alloc": cuda_peak_alloc,
        "cuda_peak_reserved": cuda_peak_resv,
    }

    if jsonl_path:
        os.makedirs(os.path.dirname(jsonl_path) or ".", exist_ok=True)
        with open(jsonl_path, "a") as f:
            f.write(json.dumps(row) + "\n")

    return row


def run_once(cfg: BenchConfig, jsonl_path: Optional[str]) -> Dict[str, Any]:
    try:
        return _train_loop(cfg, jsonl_path)
    except RuntimeError as e:
        if "out of memory" in str(e).lower() and cfg.device == "cuda":
            raise
        raise


def run_once_with_backoff(cfg: BenchConfig, jsonl_path: Optional[str], min_batch: int = 1) -> Dict[str, Any]:
    cur = cfg
    while True:
        try:
            return run_once(cur, jsonl_path)
        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" in msg and cur.device == "cuda" and cur.batch > min_batch:
                new_b = max(min_batch, cur.batch // 2)
                print(f"[bench] CUDA OOM at batch={cur.batch}; retrying with batch={new_b}")
                if torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                cur = dataclass_replace(cur, batch=new_b)
                continue
            raise


def dataclass_replace(cfg: BenchConfig, **kwargs) -> BenchConfig:
    d = asdict(cfg)
    d.update(kwargs)
    return BenchConfig(**d)


def print_table(rows, baseline_id: Optional[str] = None):
    if not rows:
        return
    # compute baseline
    base = None
    if baseline_id:
        for r in rows:
            if r["run_id"].startswith(baseline_id):
                base = r
                break
    if base is None:
        # default baseline: first c0-p0
        for r in rows:
            if r["ckpt"] == "N" and r["prec"] == "N":
                base = r
                break
    base_sps = base["samp_s"] if base else rows[0]["samp_s"]

    print("\nrun_id         device model_kind  ckpt prec fp8 avg_ms   samp/s     tok/s      loss       cuda_peak   ")
    print("-" * 102)
    for r in rows:
        peak = r.get("cuda_peak_alloc", 0)
        print(f"{r['run_id']:<13} {r['device']:<6} {r['model_kind']:<12} {r['ckpt']:<4} {r['prec']:<4} {r['fp8']:<3} "
              f"{r['avg_ms']:<8.2f} {r['samp_s']:<9.2f} {r['tok_s']:<10.2f} {r['eval_loss']:<10.4f} {human_bytes(peak):>10}")

    # speedup summary
    print()
    print("run_id         ckpt prec sdpa    avg_ms   samp/s     tok/s      eval_loss   ppl       cuda_peak    Δmem%   speedup")
    print("-" * 112)
    # compute baseline values precisely
    base_peak = base.get("cuda_peak_alloc", 0) if base else 0
    for r in rows:
        ppl = math.exp(r["eval_loss"]) if r["model_kind"] == "transformer" else float("nan")
        delta_mem = 0.0
        if base_peak > 0:
            delta_mem = 100.0 * (r.get("cuda_peak_alloc", 0) - base_peak) / base_peak
        speedup = r["samp_s"] / base_sps if base_sps > 0 else 1.0
        print(f"{r['run_id']:<13} {r['ckpt']:<4} {r['prec']:<4} {args.sdpa:<7} "
              f"{r['avg_ms']:<8.2f} {r['samp_s']:<9.2f} {r['tok_s']:<10.2f} {r['eval_loss']:<10.4f} "
              f"{ppl:<9.2f} {human_bytes(r.get('cuda_peak_alloc',0)):>10} {delta_mem:>6.2f}% {speedup:>7.3f}x")


# ------------------------------- Main ----------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DPCS microbench harness")
    parser.add_argument("--model", choices=["mlp", "transformer"], required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch", type=int, default=8)

    # MLP
    parser.add_argument("--width", type=int, default=4096)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--classes", type=int, default=1000)

    # Transformer
    parser.add_argument("--seq", type=int, default=1024)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--ff", type=int, default=2048)
    parser.add_argument("--layers", type=int, default=6)

    # training
    parser.add_argument("--optimizer", choices=["sgd", "adamw"], default="sgd")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--momentum", type=float, default=0.0)

    # policies
    parser.add_argument("--sdpa", choices=["math", "efficient", "flash"], default="math")
    parser.add_argument("--precision", choices=["on", "off", "grid"], default="off")
    parser.add_argument("--ckpt", choices=["on", "off", "grid"], default="off")
    parser.add_argument("--fp8", choices=["on", "off"], default="off")

    parser.add_argument("--jsonl", type=str, default=None)
    parser.add_argument("--grid", action="store_true", help="alias for precision=grid, ckpt=grid")

    args_local, _ = parser.parse_known_args()
    global args
    args = args_local  # used in print_table

    if args.grid:
        args.precision = "grid"
        args.ckpt = "grid"

    set_seed(1337)

    base_cfg = BenchConfig(
        model_kind=args.model,
        device=args.device,
        steps=args.steps,
        batch=args.batch,
        width=args.width,
        depth=args.depth,
        classes=args.classes,
        seq=args.seq,
        d_model=args.d_model,
        nhead=args.nhead,
        ff=args.ff,
        layers=args.layers,
        optimizer=args.optimizer,
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        sdpa=args.sdpa,
        enable_precision=(args.precision == "on"),
        enable_ckpt=(args.ckpt == "on"),
        allow_fp8=(args.fp8 == "on"),
    )

    # Expand grid
    modes = []
    prec_opts = [False, True] if args.precision == "grid" else [base_cfg.enable_precision]
    ckpt_opts = [False, True] if args.ckpt == "grid" else [base_cfg.enable_ckpt]

    for ck in ckpt_opts:
        for pr in prec_opts:
            cfg = dataclass_replace(base_cfg, enable_ckpt=ck, enable_precision=pr)
            modes.append(cfg)

    rows = []
    for cfg in modes:
        tag = ("mlp" if cfg.model_kind == "mlp" else "tra")
        c = "1" if cfg.enable_ckpt else "0"
        p = "1" if cfg.enable_precision else "0"
        run_id = f"{tag}-c{c}-p{p}-80-{cfg.optimizer}"
        r = run_once_with_backoff(cfg, args.jsonl)
        r["run_id"] = run_id
        rows.append(r)

    print_table(rows, baseline_id=("mlp-c0-p0" if args.model == "mlp" else "tra-c0-p0"))


if __name__ == "__main__":
    main()
