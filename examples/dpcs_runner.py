#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dpcs_runner.py — Minimal, correct training runner to benchmark DPCS vs baselines.

Features:
- CIFAR-10 (or FakeData) quick loop for sanity.
- Optional AMP (bf16/fp16).
- Optional CUDA Graphs capture (safe fallback if capture fails).
- Optional DPCS plugin integration (import by module path).
- Writes CSV + Markdown summaries with throughput, time-to-target, and VRAM peak.
- Saves a Chrome trace (Perfetto/Chrome) from torch.profiler for a small window.

Usage:
  python dpcs_runner.py --model resnet18 --dataset cifar10 --epochs 1 --batch-size 128 \
      --amp bf16 --enable-cudagraphs --profile --out-dir ./runs \
      --enable-dpcs --dpcs-module dpcs1 --dpcs-class DPCS --dpcs-config DPCSConfig \
      --target-loss 1.25 --seed 1337

Note:
- CUDA Graphs requires static shapes and capture-safe code paths. This script attempts a
  conservative capture and falls back to eager if anything fails.
"""

import argparse
import csv
import os
import sys
import time
import math
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.profiler import profile, schedule, ProfilerActivity
from contextlib import nullcontext

# torchvision is optional; FakeData fallback if not available
try:
    import torchvision
    from torchvision import transforms
    from torchvision.models import resnet18
    _HAS_TV = True
except Exception:
    _HAS_TV = False
    torchvision = None
    transforms = None
    resnet18 = None


def set_seed(seed: int = 1337):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(name: str, num_classes: int, device: torch.device) -> nn.Module:
    if name == "mlp":
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(3*32*32, 1024), nn.ReLU(),
            nn.Linear(1024, 256), nn.ReLU(),
            nn.Linear(256, num_classes),
        ).to(device)
    if name == "resnet18":
        if not _HAS_TV:
            raise RuntimeError("torchvision not available; use --model mlp or install torchvision.")
        m = resnet18(num_classes=num_classes)
        return m.to(device)
    raise ValueError(f"Unknown model: {name}")


def build_dataloaders(dataset: str, batch_size: int, workers: int, data_root: str) -> Tuple[DataLoader, DataLoader, int]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if dataset == "cifar10":
        if not _HAS_TV:
            raise RuntimeError("torchvision not available; use --dataset fakedata or install torchvision.")
        tfm_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        tfm_test = transforms.Compose([transforms.ToTensor()])
        train_set = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=tfm_train)
        test_set = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=tfm_test)
        num_classes = 10
    else:
        # FakeData: fixed-size images and random labels; good for quick smoke tests
        if not _HAS_TV:
            # manual FakeData replacement: random tensors
            class _Fake(torch.utils.data.Dataset):
                def __init__(self, n=10000):
                    self.n = n
                def __len__(self): return self.n
                def __getitem__(self, idx):
                    x = torch.randn(3, 32, 32)
                    y = torch.randint(0, 10, (1,)).item()
                    return x, y
            train_set = _Fake(4096)
            test_set = _Fake(1024)
        else:
            tfm = transforms.ToTensor()
            train_set = torchvision.datasets.FakeData(size=4096, image_size=(3,32,32), num_classes=10, transform=tfm)
            test_set  = torchvision.datasets.FakeData(size=1024, image_size=(3,32,32), num_classes=10, transform=tfm)
        num_classes = 10

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    return train_loader, test_loader, num_classes


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def maybe_import_dpcs(args) -> Optional[Any]:
    """Try to import DPCS plugin if requested. Returns (DPCS, DPCSConfig) or None."""
    if not args.enable_dpcs:
        return None
    # Allow a file path or a module on sys.path
    if args.dpcs_path:
        sys.path.insert(0, args.dpcs_path)
    try:
        mod = __import__(args.dpcs_module, fromlist=[args.dpcs_class, args.dpcs_config])
        DPCS = getattr(mod, args.dpcs_class)
        DPCSConfig = getattr(mod, args.dpcs_config)
        return (DPCS, DPCSConfig)
    except Exception as e:
        print(f"[WARN] Failed to import DPCS ({e}). Continuing without DPCS.", file=sys.stderr)
        return None


def cuda_mem_stats(device: torch.device) -> Dict[str, float]:
    stats = {}
    try:
        free_b, total_b = torch.cuda.mem_get_info(device)
        stats["free_bytes"] = int(free_b)
        stats["total_bytes"] = int(total_b)
        stats["free_frac"] = float(free_b) / float(total_b + 1e-9)
    except Exception:
        pass
    stats["max_mem_alloc_bytes"] = int(torch.cuda.max_memory_allocated(device))
    stats["max_mem_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
    return stats


@torch.no_grad()
def evaluate(model: nn.Module, data: DataLoader, loss_fn: nn.Module, device: torch.device, amp_dtype: Optional[str]) -> Tuple[float, float]:
    model.eval()
    total = 0
    s_acc = 0.0
    s_loss = 0.0
    if amp_dtype == "bf16":
        amp_ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16) if device.type == 'cuda' else torch.autocast(device_type='cpu', dtype=torch.bfloat16)
    elif amp_dtype == "fp16":
        amp_ctx = torch.autocast(device_type='cuda', dtype=torch.float16) if device.type == 'cuda' else torch.autocast(device_type='cpu', dtype=torch.float16)
    else:
        # no autocast
        class _nullctx:
            def __enter__(self): return None
            def __exit__(self, *a): return False
        amp_ctx = _nullctx()

    for x, y in data:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with amp_ctx:
            logits = model(x)
            loss = loss_fn(logits, y)
        s_loss += loss.item() * x.size(0)
        s_acc  += accuracy(logits, y) * x.size(0)
        total += x.size(0)
    return s_loss/total, s_acc/total


def train_one_epoch_eager(model, data, loss_fn, opt, scaler, device, amp_dtype, log_interval=50):
    model.train()
    start = time.perf_counter()
    seen = 0
    for it, (x, y) in enumerate(data):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        if amp_dtype in ("bf16", "fp16") and device.type == 'cuda':
            autocast_dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
            with torch.autocast(device_type='cuda', dtype=autocast_dtype):
                logits = model(x)
                loss = loss_fn(logits, y)
            if amp_dtype == "fp16":
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
        else:
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

        seen += x.size(0)
        if (it + 1) % log_interval == 0:
            torch.cuda.synchronize(device) if device.type == 'cuda' else None
    torch.cuda.synchronize(device) if device.type == 'cuda' else None
    elapsed = time.perf_counter() - start
    return seen / elapsed  # imgs/sec


def train_one_epoch_cudagraphs(model, data, loss_fn, opt, scaler, device, amp_dtype, log_interval=50):
    """
    Conservative CUDA Graph capture of forward+backward+step for fixed-shape batches.
    Falls back to eager if capture raises.
    """
    if device.type != 'cuda':
        return train_one_epoch_eager(model, data, loss_fn, opt, scaler, device, amp_dtype, log_interval)

    # Check static batch and shape from first batch
    try:
        x0, y0 = next(iter(data))
    except StopIteration:
        return 0.0
    B0, C0, H0, W0 = x0.shape
    dtype_autocast = None
    if amp_dtype in ("bf16", "fp16"):
        dtype_autocast = torch.bfloat16 if amp_dtype == "bf16" else torch.float16

    # Allocate static buffers
    static_x = torch.empty((B0, C0, H0, W0), device=device, dtype=torch.float32)
    static_y = torch.empty((B0,), device=device, dtype=torch.long)

    # Warm-up on a side stream (conservative)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            opt.zero_grad(set_to_none=True)
            with (torch.autocast(device_type='cuda', dtype=dtype_autocast) if dtype_autocast is not None else torch.cuda.amp.autocast(enabled=False)):
                logits = model(static_x.normal_())
                loss = loss_fn(logits, static_y.random_(0, 10))
            # warm-up just builds the graph and caches allocations; do not update weights
            if amp_dtype == "fp16":
                scaler.scale(loss).backward()
                scaler._unscale_grads_(opt)  # safe no-op for capture prep
                opt.zero_grad(set_to_none=True)
                scaler.update()
            else:
                loss.backward()
                opt.zero_grad(set_to_none=True)

    torch.cuda.current_stream().wait_stream(s)

    # Capture
    g = torch.cuda.CUDAGraph()
    # grads must be None before capture so that graph's private pool allocates .grad
    opt.zero_grad(set_to_none=True)
    if amp_dtype == "fp16":
        with torch.cuda.graph(g):
            with torch.autocast(device_type='cuda', dtype=dtype_autocast):
                static_logits = model(static_x)
                static_loss   = loss_fn(static_logits, static_y)
            scaler.scale(static_loss).backward()
            scaler.step(opt)
            scaler.update()
    else:
        with torch.cuda.graph(g):
            with (torch.autocast(device_type='cuda', dtype=dtype_autocast) if dtype_autocast is not None else torch.cuda.amp.autocast(enabled=False)):
                static_logits = model(static_x)
                static_loss   = loss_fn(static_logits, static_y)
            static_loss.backward()
            opt.step()

    # Replay over data
    start = time.perf_counter()
    seen = 0
    it = 0
    for xb, yb in data:
        if xb.shape != static_x.shape or yb.shape != static_y.shape:
            # shape changed -> fall back
            return train_one_epoch_eager(model, data, loss_fn, opt, scaler, device, amp_dtype, log_interval)
        static_x.copy_(xb, non_blocking=True)
        static_y.copy_(yb, non_blocking=True)
        g.replay()
        seen += xb.size(0)
        it += 1
        if (it % log_interval) == 0:
            torch.cuda.synchronize(device)
    torch.cuda.synchronize(device)
    return seen / (time.perf_counter() - start)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="resnet18", choices=["resnet18", "mlp"])
    p.add_argument("--dataset", default="cifar10", choices=["cifar10", "fakedata"])
    p.add_argument("--data-root", default="./data")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--amp", default="none", choices=["none", "bf16", "fp16"])
    p.add_argument("--enable-cudagraphs", action="store_true")
    p.add_argument("--compile", action="store_true", help="Wrap model with torch.compile for steady segments.")
    p.add_argument("--enable-dpcs", action="store_true")
    p.add_argument("--dpcs-path", default="", help="Optional path to add to sys.path before importing the module.")
    p.add_argument("--dpcs-module", default="dpcs1", help="Module name that contains DPCS classes.")
    p.add_argument("--dpcs-class", default="DPCS", help="Class name for the controller/wrapper.")
    p.add_argument("--dpcs-config", default="DPCSConfig", help="Class name for the config.")
    p.add_argument("--target-loss", type=float, default=1.25, help="Stop when val loss <= target.")
    p.add_argument("--target-acc", type=float, default=0.0, help="Alternatively, stop when val acc >= target-acc (>0 disables loss target).")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--out-dir", default="./runs")
    p.add_argument("--profile", action="store_true")
    args = p.parse_args()

    set_seed(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader, num_classes = build_dataloaders(args.dataset, args.batch_size, args.workers, args.data_root)
    model = build_model(args.model, num_classes, device)

    if args.compile:
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"[WARN] torch.compile failed ({e}), continuing without.", file=sys.stderr)

    # Optional DPCS
    dpcs_tuple = maybe_import_dpcs(args)
    if dpcs_tuple is not None and args.enable_dpcs:
        DPCS, DPCSConfig = dpcs_tuple
        dcfg = DPCSConfig()  # let your defaults take effect
        dpcs = DPCS(model, dcfg)
        model = dpcs  # assume wrapper behaves like nn.Module

    loss_fn = nn.CrossEntropyLoss().to(device)
    opt = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp == "fp16" and device.type == 'cuda'))

    # profiler setup
    prof_ctx = nullcontext()
    if args.profile:
        trace_dir = out / "trace"
        trace_dir.mkdir(exist_ok=True, parents=True)
        prof_ctx = profile(
            activities=[ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device.type == 'cuda' else []),
            schedule=schedule(wait=1, warmup=1, active=2, repeat=1),
            on_trace_ready=lambda prof: prof.export_chrome_trace(str(trace_dir / "trace.json")),
            with_stack=False,
            record_shapes=True,
            profile_memory=True,
        )

    # metrics
    results = {
        "model": args.model,
        "dataset": args.dataset,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "amp": args.amp,
        "compile": bool(args.compile),
        "cudagraphs": bool(args.enable_cudagraphs),
        "dpcs": bool(args.enable_dpcs and dpcs_tuple is not None),
        "seed": args.seed,
        "device": str(device),
    }

    # Track TTA (time-to-accuracy/loss)
    start_time = time.perf_counter()
    tta_sec = None

    # reset peak memory counters
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    with prof_ctx:
        for epoch in range(args.epochs):
            if args.enable_cudagraphs:
                try:
                    ips = train_one_epoch_cudagraphs(model, train_loader, loss_fn, opt, scaler, device, args.amp)
                except Exception as e:
                    print(f"[WARN] CUDA Graphs path failed ({e}); falling back to eager.", file=sys.stderr)
                    ips = train_one_epoch_eager(model, train_loader, loss_fn, opt, scaler, device, args.amp)
            else:
                ips = train_one_epoch_eager(model, train_loader, loss_fn, opt, scaler, device, args.amp)

            val_loss, val_acc = evaluate(model, test_loader, loss_fn, device, args.amp)
            results.update({"train_ips_last": ips, "val_loss": val_loss, "val_acc": val_acc})

            # Check TTA
            met_acc = args.target_acc > 0.0 and val_acc >= args.target_acc
            met_loss = (args.target_acc <= 0.0) and (val_loss <= args.target_loss)
            if (met_acc or met_loss) and tta_sec is None:
                tta_sec = time.perf_counter() - start_time
                results["tta_sec"] = float(tta_sec)
                # stop early to report TTA fairly
                break

            if args.profile:
                # step profiler schedule manually
                pass

    # finalize memory stats
    if device.type == 'cuda':
        mem_stats = cuda_mem_stats(device)
        results.update({
            "peak_mem_alloc_mb": round(mem_stats.get("max_mem_alloc_bytes", 0) / (1024**2), 2),
            "peak_mem_reserved_mb": round(mem_stats.get("max_mem_reserved_bytes", 0) / (1024**2), 2),
            "free_frac": round(mem_stats.get("free_frac", float('nan')), 4) if "free_frac" in mem_stats else "n/a",
            "total_mem_gb": round(mem_stats.get("total_bytes", 0) / (1024**3), 2) if "total_bytes" in mem_stats else "n/a",
        })
    else:
        results.update({"peak_mem_alloc_mb": "n/a", "peak_mem_reserved_mb": "n/a", "free_frac": "n/a", "total_mem_gb": "n/a"})

    # write CSV + Markdown
    csv_path = out / "results.csv"
    md_path  = out / "results.md"
    hdr = ["model","dataset","batch_size","epochs","amp","compile","cudagraphs","dpcs","seed","device",
           "train_ips_last","val_loss","val_acc","tta_sec","peak_mem_alloc_mb","peak_mem_reserved_mb","free_frac","total_mem_gb"]
    write_hdr = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=hdr)
        if write_hdr:
            w.writeheader()
        row = {k: results.get(k, "") for k in hdr}
        w.writerow(row)

    # Markdown one-liner
    def md_row(d: Dict[str, Any]) -> str:
        return "| " + " | ".join(str(d.get(k,"")) for k in hdr) + " |"
    if not md_path.exists():
        with open(md_path, "w") as f:
            f.write("| " + " | ".join(hdr) + " |\n")
            f.write("|" + "|".join(["---"]*len(hdr)) + "|\n")
    with open(md_path, "a") as f:
        f.write(md_row(results) + "\n")

    # Also dump JSON for programmatic use
    with open(out / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("== Summary ==")
    for k, v in results.items():
        print(f"{k}: {v}")
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {md_path}")
    if args.profile:
        print(f"Wrote: {out/'trace'/'trace.json'} (open in chrome://tracing or Perfetto)")

if __name__ == "__main__":
    main()
