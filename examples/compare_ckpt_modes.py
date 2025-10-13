#!/usr/bin/env python3
"""Compare DPCS per-leaf checkpointing vs. PyTorch selective checkpointing."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from dpcs import DPCS


class ToyLM(nn.Module):
    """Tiny Transformer-like encoder for checkpointing experiments."""

    def __init__(self, hidden: int, heads: int, ff: int, layers: int, vocab: int) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab, hidden)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=ff,
            batch_first=True,
            dropout=0.0,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.lm_head = nn.Linear(hidden, vocab)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        h = self.emb(tokens)
        h = self.encoder(h)
        return self.lm_head(h)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__ or "compare checkpointing modes")
    parser.add_argument("--output", type=Path, default=Path("ckpt_compare.jsonl"), help="JSONL output path")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--hidden", type=int, default=512, help="Hidden size")
    parser.add_argument("--heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--ff", type=int, default=2048, help="Feed-forward width")
    parser.add_argument("--layers", type=int, default=6, help="Number of encoder layers")
    parser.add_argument("--vocab", type=int, default=30522, help="Vocabulary size")
    parser.add_argument("--seq", type=int, nargs="*", default=[256, 512, 1024, 1536])
    parser.add_argument("--steps", type=int, default=4, help="Timed steps per configuration")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup steps before timing")
    parser.add_argument(
        "--budget-frac",
        type=float,
        default=0.5,
        help="Activation memory budget fraction for delegate mode",
    )
    return parser.parse_args()


def is_oom(err: RuntimeError) -> bool:
    msg = str(err).lower()
    return "out of memory" in msg or "cuda error" in msg


def run_mode(
    mode: str,
    seq_len: int,
    args: argparse.Namespace,
) -> Dict[str, object]:
    device = args.device
    model = ToyLM(args.hidden, args.heads, args.ff, args.layers, args.vocab).to(device).train()
    scheduler_kwargs = dict(device_type=device, enable_precision=False)
    if mode == "delegate":
        scheduler_kwargs.update(
            delegate_selective_ckpt=True,
            activation_memory_budget_frac=args.budget_frac,
        )
    sched = DPCS(**scheduler_kwargs)
    model = sched.wrap(model)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    def step() -> int:
        sched.start_step()
        opt.zero_grad(set_to_none=True)
        tokens = torch.randint(args.vocab, (args.batch, seq_len), device=device)
        targets = torch.randint(args.vocab, (args.batch, seq_len), device=device)
        with sched.forward_context():
            logits = model(tokens)
        loss = F.cross_entropy(logits.view(-1, args.vocab), targets.view(-1))
        loss.backward()
        sched.collect_signals(loss, model)
        opt.step()
        sched.end_step(opt)
        return tokens.numel()

    oom_count = 0

    try:
        for _ in range(args.warmup):
            step()
    except RuntimeError as err:  # warmup OOM -> bail early
        if is_oom(err):
            oom_count += 1
            if device == "cuda":
                torch.cuda.empty_cache()
            return {
                "mode": mode,
                "seq_len": seq_len,
                "tokens_per_s": 0.0,
                "duration_s": 0.0,
                "tokens": 0,
                "peak_mem_bytes": None,
                "ooms": oom_count,
            }
        raise

    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    tokens_total = 0
    start = time.perf_counter()
    try:
        for _ in range(args.steps):
            tokens_total += step()
    except RuntimeError as err:
        if is_oom(err):
            oom_count += 1
        else:
            raise
    end = time.perf_counter()

    if device == "cuda":
        torch.cuda.synchronize()
        peak_mem = torch.cuda.max_memory_allocated()
    else:
        peak_mem = None

    duration = max(0.0, end - start)
    tokens_per_s = tokens_total / duration if duration > 0 else 0.0

    return {
        "mode": mode,
        "seq_len": seq_len,
        "tokens_per_s": tokens_per_s,
        "duration_s": duration,
        "tokens": tokens_total,
        "peak_mem_bytes": peak_mem,
        "ooms": oom_count,
    }


def main() -> None:
    args = parse_args()
    results: List[Dict[str, object]] = []
    for seq_len in args.seq:
        for mode in ("dpcs", "delegate"):
            metrics = run_mode(mode, seq_len, args)
            results.append(metrics)
            print(json.dumps(metrics, indent=2))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as fh:
            for record in results:
                fh.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
