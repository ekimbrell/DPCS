"""Static AMP + checkpointing DPCS benchmark.

Runs a tiny Transformer for a handful of steps with:
- static AMP (forced bf16/fp16 on CUDA),
- static activation checkpointing (all leaves checkpointed),
- DPCS enabled for runtime instrumentation.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if SRC_DIR.exists():
    src_path = str(SRC_DIR)
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

from dpcs import DPCS  # noqa: E402
from dpcs.runtime import (
    amp_preferred_dtype,
    get_step_peak,
)


class TinyTransformer(nn.Module):
    """Minimal Transformer encoder + classifier head."""

    def __init__(
        self,
        vocab: int,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        layers: int,
    ) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.embed = nn.Embedding(vocab, embed_dim)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens)
        x = self.encoder(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.head(x)


@dataclass
class BenchConfig:
    steps: int
    warmup: int
    batch_size: int
    seq_len: int
    vocab: int
    embed_dim: int
    num_heads: int
    ff_dim: int
    layers: int
    seed: int
    device: torch.device


def _auto_device(value: str) -> torch.device:
    if value and value.lower() != "auto":
        return torch.device(value)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parse_args() -> BenchConfig:
    parser = argparse.ArgumentParser(description="Static AMP + checkpointing DPCS benchmark")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--vocab", type=int, default=128)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--ff-dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    return BenchConfig(
        steps=max(1, args.steps),
        warmup=max(0, args.warmup),
        batch_size=max(1, args.batch_size),
        seq_len=max(1, args.seq_len),
        vocab=max(8, args.vocab),
        embed_dim=max(8, args.embed_dim),
        num_heads=max(1, args.num_heads),
        ff_dim=max(16, args.ff_dim),
        layers=max(1, args.layers),
        seed=args.seed,
        device=_auto_device(args.device),
    )


def _build_dpcs(device: torch.device) -> Tuple[DPCS, Optional[torch.amp.GradScaler]]:
    scheduler = DPCS(
        device_type=device.type,
        enable_precision=1,
        curv_period=0,
        ckpt_topk_frac=1.0,
        min_activation_bytes_to_ckpt=0,
        ckpt_use_benefit_score=0,
        ckpt_patience=1,
        checkpoint_cfg={"max_fraction": 1.0},
    )
    scheduler.enable_checkpointing(True)
    scheduler.vram_headroom = lambda: 0.0

    if device.type == "cuda" and torch.cuda.is_available():
        pref_dtype = amp_preferred_dtype("cuda")
        if pref_dtype is torch.bfloat16:
            scheduler.force_precision("bf16")
        else:
            scheduler.force_precision("fp16")
    else:
        scheduler.force_precision("fp32")

    dev_type, dtype, enabled = scheduler.get_amp_config()
    use_scaler = enabled and scheduler.amp_uses_grad_scaler()
    if use_scaler:
        scaler = torch.amp.GradScaler(enabled=True, init_scale=2.0**8)
    else:
        scaler = None
    return scheduler, scaler


def main() -> None:
    cfg = _parse_args()
    _seed_everything(cfg.seed)

    device = cfg.device
    model = TinyTransformer(
        vocab=cfg.vocab,
        embed_dim=cfg.embed_dim,
        num_heads=cfg.num_heads,
        ff_dim=cfg.ff_dim,
        layers=cfg.layers,
    ).to(device)

    scheduler, scaler = _build_dpcs(device)
    model = scheduler.wrap(model)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    dev_type, dtype, amp_enabled = scheduler.get_amp_config()
    overflow_monitor = scheduler.overflow_monitor(scaler) if scaler is not None else None

    total_steps = cfg.steps + cfg.warmup
    measured_steps = 0
    overflow_count = 0
    peak_bytes = 0

    def _sync() -> None:
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device)

    _sync()
    start_time = time.perf_counter()

    for step in range(total_steps):
        scheduler.start_step()
        optimizer.zero_grad(set_to_none=True)

        tokens = torch.randint(0, cfg.vocab, (cfg.batch_size, cfg.seq_len), device=device)
        targets = torch.randint(0, cfg.vocab, (cfg.batch_size,), device=device)

        ctx = torch.autocast(device_type=dev_type, dtype=dtype, enabled=amp_enabled)
        with scheduler.forward_context(), ctx:
            logits = model(tokens)
            loss = F.cross_entropy(logits, targets)

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        scheduler.collect_signals(loss, model)

        if scaler is not None:
            scaler.step(optimizer)
            scheduler.end_step(optimizer, scaler)
            with overflow_monitor:  # type: ignore[union-attr]
                scaler.update()
            if overflow_monitor.last_overflow:  # type: ignore[union-attr]
                overflow_count += 1
        else:
            optimizer.step()
            scheduler.end_step(optimizer, None)

        step_peak = int(get_step_peak(device))
        peak_bytes = max(peak_bytes, step_peak)

        if step + 1 == cfg.warmup:
            _sync()
            start_time = time.perf_counter()
            measured_steps = 0
        if step + 1 > cfg.warmup:
            measured_steps += 1

    _sync()
    elapsed = max(time.perf_counter() - start_time, 1e-8)
    steps_per_sec = measured_steps / elapsed if measured_steps else 0.0

    result = {
        "device": str(device),
        "steps": measured_steps,
        "steps_per_sec": steps_per_sec,
        "peak_memory_bytes": peak_bytes,
        "overflow_count": overflow_count,
        "amp_enabled": amp_enabled,
        "amp_dtype": str(dtype),
        "checkpointing": True,
    }

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
