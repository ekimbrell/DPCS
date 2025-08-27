#!/usr/bin/env python3
"""
DPCS micro-benchmark harness

Runs a small grid over:
  - checkpointing:       off / on
  - precision scheduling: off / on
  - FP8 (Transformer Engine): off / on (only if available)

Collects per-run metrics:
  - avg_step_ms, samples_per_s (and tokens_per_s for Transformer)
  - final loss
  - CUDA peak allocator bytes (if on CUDA)
  - precision mix summary from DPCS
  - overflow hints (GradScaler scale drops)

Outputs JSONL (one line per run) and prints a compact table.

Examples
--------
# MLP on CUDA, 24 steps, large batch, JSONL to results.jsonl
python bench.py --model mlp --device cuda --steps 24 --batch 64 \
  --width 8192 --depth 10 --classes 1000 --jsonl results.jsonl

# Transformer on CUDA, force MATH SDPA for clearer checkpoint memory deltas
python bench.py --model transformer --device cuda --steps 20 --batch 8 \
  --seq 1024 --d-model 512 --nhead 8 --ff 2048 --layers 6 --force-math-sdpa
"""
import argparse
import json
import math
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from dpcs import DPCS

# -------- SDPA backend control (version-portable) ----------------------------

def make_force_math_sdpa():
    try:
        from torch.nn.attention import sdpa_kernel, SDPBackend  # PyTorch ≥2.3ish

        def _ctx():
            return sdpa_kernel([SDPBackend.MATH])

        return _ctx
    except Exception:
        try:
            from torch.backends.cuda import sdp_kernel as _old  # deprecated API

            def _ctx():
                return _old(enable_flash=False, enable_mem_efficient=False, enable_math=True)

            return _ctx
        except Exception:
            def _ctx():
                return nullcontext()

            return _ctx


# -------- Models -------------------------------------------------------------

class DeepMLP(nn.Module):
    """Activation-heavy MLP to magnify checkpointing effects."""
    def __init__(self, width=8192, depth=10, out_dim=1000):
        super().__init__()
        layers: List[nn.Module] = []
        layers.append(nn.Linear(width, width))
        for _ in range(depth - 1):
            layers += [nn.GELU(), nn.Linear(width, width)]
        layers += [nn.GELU(), nn.Linear(width, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TinyTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, dim_ff=2048, nlayers=6, n_classes=1000, dropout=0.0):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=nlayers)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x):  # x: [B, S, D]
        h = self.enc(x)
        h = h.mean(dim=1)
        return self.head(h)


# -------- Config & results structures ---------------------------------------

@dataclass
class RunConfig:
    device: str
    model_kind: str
    steps: int
    batch: int
    seq: Optional[int] = None
    d_model: Optional[int] = None
    nhead: Optional[int] = None
    dim_ff: Optional[int] = None
    layers: Optional[int] = None
    width: Optional[int] = None
    depth: Optional[int] = None
    classes: int = 1000

    ckpt_on: bool = False
    enable_precision: bool = False
    allow_fp8: bool = False

    force_math_sdpa: bool = False
    min_ckpt_bytes: int = 8 << 20


@dataclass
class RunResult:
    run_id: str
    device: str
    model_kind: str
    steps: int
    batch: int
    seq: Optional[int]
    d_model: Optional[int]
    nhead: Optional[int]
    dim_ff: Optional[int]
    layers: Optional[int]
    width: Optional[int]
    depth: Optional[int]
    classes: int

    ckpt_on: bool
    enable_precision: bool
    allow_fp8: bool
    precision_mix: Dict[str, int]

    avg_step_ms: float
    samples_per_s: float
    tokens_per_s: Optional[float]
    loss_last: float
    cuda_peak_bytes: Optional[int]


# -------- Utilities ----------------------------------------------------------

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _num_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def _device_select(arg: str) -> str:
    if arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return arg


def _pretty_bytes(n: Optional[int]) -> str:
    if n is None:
        return "-"
    # kib / mib / gib
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    x = float(n)
    i = 0
    while x >= 1024.0 and i < len(units) - 1:
        x /= 1024.0
        i += 1
    return f"{x:.2f} {units[i]}"


def _print_table(rows: List[RunResult]) -> None:
    cols = [
        ("run_id", 9), ("device", 5), ("model_kind", 11), ("ckpt", 4), ("prec", 4), ("fp8", 3),
        ("avg_ms", 8), ("samp/s", 10), ("tok/s", 10), ("loss", 10), ("cuda_peak", 12)
    ]
    header = " ".join(name.ljust(w) for name, w in cols)
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        values = {
            "run_id": r.run_id,
            "device": r.device,
            "model_kind": r.model_kind,
            "ckpt": "Y" if r.ckpt_on else "N",
            "prec": "Y" if r.enable_precision else "N",
            "fp8": "Y" if r.allow_fp8 else "N",
            "avg_ms": f"{r.avg_step_ms:.2f}",
            "samp/s": f"{r.samples_per_s:.2f}",
            "tok/s": f"{0.0 if r.tokens_per_s is None else r.tokens_per_s:.2f}",
            "loss": f"{r.loss_last:.4f}",
            "cuda_peak": _pretty_bytes(r.cuda_peak_bytes),
        }
        print(" ".join(str(values[name]).ljust(w) for name, w in cols))


# -------- Core bench ---------------------------------------------------------

def build_model(cfg: RunConfig) -> Tuple[nn.Module, torch.Tensor, Optional[torch.Tensor]]:
    device = cfg.device
    if cfg.model_kind == "mlp":
        width = cfg.width or 8192
        depth = cfg.depth or 10
        classes = cfg.classes
        model = DeepMLP(width=width, depth=depth, out_dim=classes).to(device)
        x = torch.randn(cfg.batch, width, device=device)
        y = torch.randint(0, classes, (cfg.batch,), device=device)
        return model, x, y
    elif cfg.model_kind == "transformer":
        D = cfg.d_model or 512
        H = cfg.nhead or 8
        FF = cfg.dim_ff or 2048
        L = cfg.layers or 6
        S = cfg.seq or 1024
        C = cfg.classes
        model = TinyTransformer(d_model=D, nhead=H, dim_ff=FF, nlayers=L, n_classes=C, dropout=0.0).to(device)
        x = torch.randn(cfg.batch, S, D, device=device)
        y = torch.randint(0, C, (cfg.batch,), device=device)
        return model, x, y
    else:
        raise ValueError(f"Unknown model_kind: {cfg.model_kind}")


def run_once(cfg: RunConfig, jsonl: Optional[str]) -> RunResult:
    device = cfg.device
    force_math_sdpa = make_force_math_sdpa()

    model, x, y = build_model(cfg)

    # Scheduler
    dpcs = DPCS(
        device_type=device,
        enable_precision=cfg.enable_precision,
        allow_fp8=cfg.allow_fp8,
        wrap_types=(nn.Linear, nn.TransformerEncoderLayer),
        min_activation_bytes_to_ckpt=cfg.min_ckpt_bytes,
    )
    model = dpcs.wrap(model)

    # Optimizer & (optional) GradScaler for CUDA when precision is on
    opt = optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda" and cfg.enable_precision))
    crit = nn.CrossEntropyLoss().to(device)

    # Control checkpointing explicitly for the whole run
    dpcs._ckpt_on = bool(cfg.ckpt_on)

    # Timings
    step_times: List[float] = []

    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    tokens_per_step: Optional[int] = None
    if cfg.model_kind == "transformer":
        tokens_per_step = cfg.batch * (cfg.seq or 1024)

    # A per-run SDPA context if requested and device is CUDA
    sdpa_ctx = force_math_sdpa() if (cfg.force_math_sdpa and device == "cuda" and cfg.model_kind == "transformer") else nullcontext()

    last_loss = float("nan")

    with sdpa_ctx:
        for step in range(cfg.steps):
            t0 = time.perf_counter()

            dpcs.start_step()

            # forward
            if scaler.is_enabled():
                with torch.cuda.amp.autocast():
                    logits = model(x)
                    loss = crit(logits, y)
            else:
                logits = model(x)
                loss = crit(logits, y)

            # backward
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # collect signals with access to grads
            dpcs.collect_signals(loss, model)

            # step
            if scaler.is_enabled():
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            opt.zero_grad(set_to_none=True)

            # end scheduler step (for decisions, cooldown, logging)
            dpcs.end_step(opt, scaler if scaler.is_enabled() else None)

            if device == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            step_times.append((t1 - t0) * 1000.0)  # ms
            last_loss = float(loss.item())

    avg_ms = sum(step_times) / max(1, len(step_times))
    samples_per_s = 1000.0 * cfg.batch / avg_ms
    toks_per_s = (1000.0 * tokens_per_step / avg_ms) if tokens_per_step is not None else None

    cuda_peak = int(torch.cuda.max_memory_allocated(device)) if device == "cuda" else None

    result = RunResult(
        run_id=f"{cfg.model_kind[:3]}-c{int(cfg.ckpt_on)}-p{int(cfg.enable_precision)}-8{int(cfg.allow_fp8)}",
        device=device,
        model_kind=cfg.model_kind,
        steps=cfg.steps,
        batch=cfg.batch,
        seq=cfg.seq if cfg.model_kind == "transformer" else None,
        d_model=cfg.d_model if cfg.model_kind == "transformer" else None,
        nhead=cfg.nhead if cfg.model_kind == "transformer" else None,
        dim_ff=cfg.dim_ff if cfg.model_kind == "transformer" else None,
        layers=cfg.layers if cfg.model_kind == "transformer" else None,
        width=cfg.width if cfg.model_kind == "mlp" else None,
        depth=cfg.depth if cfg.model_kind == "mlp" else None,
        classes=cfg.classes,
        ckpt_on=cfg.ckpt_on,
        enable_precision=cfg.enable_precision,
        allow_fp8=cfg.allow_fp8,
        precision_mix=dpcs.modes_summary(),
        avg_step_ms=avg_ms,
        samples_per_s=samples_per_s,
        tokens_per_s=toks_per_s,
        loss_last=last_loss,
        cuda_peak_bytes=cuda_peak,
    )

    if jsonl:
        with open(jsonl, "a", buffering=1) as f:
            f.write(json.dumps(asdict(result)) + "\n")

    return result


# -------- Main ---------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="DPCS micro-benchmark harness")
    p.add_argument("--model", choices=["mlp", "transformer"], default="transformer")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--batch", type=int, default=8)

    # Transformer-only sizes
    p.add_argument("--seq", type=int, default=1024)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--ff", type=int, default=2048)
    p.add_argument("--layers", type=int, default=6)

    # MLP-only sizes
    p.add_argument("--width", type=int, default=8192)
    p.add_argument("--depth", type=int, default=10)

    # General
    p.add_argument("--classes", type=int, default=1000)
    p.add_argument("--jsonl", type=str, default="bench_results.jsonl")
    p.add_argument("--seed", type=int, default=123)

    # DPCS behavior
    p.add_argument("--force-math-sdpa", action="store_true", help="Force math SDPA (transformer+CUDA)")
    p.add_argument("--min-ckpt-bytes", type=int, default=(8 << 20))

    # Grid control (by default runs combinations)
    p.add_argument("--ckpt", choices=["grid", "off", "on"], default="grid")
    p.add_argument("--precision", choices=["grid", "off", "on"], default="grid")
    p.add_argument("--fp8", choices=["grid", "off", "on"], default="grid")

    args = p.parse_args()

    set_seed(args.seed)
    device = _device_select(args.device)

    # Build base RunConfig
    base = RunConfig(
        device=device,
        model_kind=args.model,
        steps=args.steps,
        batch=args.batch,
        seq=args.seq,
        d_model=args.d_model,
        nhead=args.nhead,
        dim_ff=args.ff,
        layers=args.layers,
        width=args.width,
        depth=args.depth,
        classes=args.classes,
        force_math_sdpa=bool(args.force_math_sdpa),
        min_ckpt_bytes=int(args.min_ckpt_bytes),
    )

    # Enumerate grid
    def expand(opt: str) -> List[int]:
        return [0, 1] if opt == "grid" else ([1] if opt == "on" else [0])

    ckpts = expand(args.ckpt)
    precs = expand(args.precision)
    fp8s = expand(args.fp8)

    results: List[RunResult] = []

    # Capability check for FP8
    te_available = False
    try:
        import transformer_engine.pytorch as _te  # noqa: F401
        te_available = True
    except Exception:
        te_available = False

    for ck in ckpts:
        for pr in precs:
            for f8 in fp8s:
                if f8 and not te_available:
                    print("[skip] FP8 run requested but Transformer Engine is not available.")
                    continue
                cfg = base
                cfg = RunConfig(**{**cfg.__dict__, "ckpt_on": bool(ck), "enable_precision": bool(pr), "allow_fp8": bool(f8)})
                r = run_once(cfg, args.jsonl)
                results.append(r)

    _print_table(results)


if __name__ == "__main__":
    main()
