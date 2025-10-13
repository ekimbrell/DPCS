"""DPCS micro-benchmark suite.

This harness executes a tiny Transformer on random data across a grid of
configuration toggles to measure Dynamic Precision & Checkpointing Scheduler
(DPCS) overhead. For every combination the script records throughput and
latency statistics together with runtime counters such as AMP overflow events
and bytes kept under activation checkpointing. Results are written to a JSONL
file – one object per configuration – for easy downstream analysis.

The toggles covered by default are:

* Grad/curvature signals enabled vs. disabled.
* Curvature probe cadence (0 disables probes).
* Activation checkpointing disabled vs. automatic planning.
* AMP precision control: fixed (manual) vs. dynamic (driven by DPCS).
* TransformerEngine FP8 disabled vs. enabled (when available).

Optionally a ``torch.profiler`` capture can be emitted for a chosen
combination, either as a Chrome trace or as a top-level table summarising
Python vs. kernel time.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.optim as optim


ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if SRC_DIR.exists():
    src_path = str(SRC_DIR)
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

from dpcs import DPCS  # noqa: E402  (lazy path insertion above)


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------


class TransformerBlock(nn.Module):
    """Minimal Transformer block used in the benchmark."""

    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=0.0, batch_first=True
        )
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            OrderedDict(
                [
                    ("fc1", nn.Linear(embed_dim, ff_dim)),
                    ("act", nn.GELU()),
                    ("fc2", nn.Linear(ff_dim, embed_dim)),
                ]
            )
        )
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.ln1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.ln2(x + ffn_out)
        return x


class TinyTransformer(nn.Module):
    """Tiny Transformer with embedding + pooling head."""

    def __init__(
        self,
        vocab: int = 128,
        embed_dim: int = 128,
        num_heads: int = 4,
        ff_dim: int = 256,
        layers: int = 2,
    ) -> None:
        super().__init__()

        self.embed = nn.Embedding(vocab, embed_dim)
        blocks = []
        for i in range(layers):
            blocks.append((f"block{i}", TransformerBlock(embed_dim, num_heads, ff_dim)))
        self.blocks = nn.Sequential(OrderedDict(blocks))
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens)
        x = self.blocks(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.head(x)


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------


def _auto_device(arg: str) -> torch.device:
    if arg and arg.lower() != "auto":
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    q = min(1.0, max(0.0, float(q)))
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    idx = q * (len(ordered) - 1)
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return float(ordered[int(idx)])
    weight = idx - lo
    return float((1.0 - weight) * ordered[lo] + weight * ordered[hi])


@dataclass
class Toggle:
    signals: str
    curv_period: int
    checkpoint: str
    amp_mode: str
    fp8: str


@dataclass
class ProfileCfg:
    enabled: bool
    kind: str  # none|table|trace|both
    out_dir: Path


def _expand_toggles(args: argparse.Namespace) -> List[Toggle]:
    signals = args.signals or ["on", "off"]
    curv = args.curv_periods or [0, 16]
    ckpt = args.checkpoint_modes or ["off", "auto"]
    amp = args.amp_modes or ["fixed", "dynamic"]
    fp8 = args.fp8 or ["off", "on"]

    combos: List[Toggle] = []
    for s in signals:
        for cp in curv:
            for ck in ckpt:
                for am in amp:
                    for fp in fp8:
                        combos.append(
                            Toggle(
                                signals=s,
                                curv_period=int(cp),
                                checkpoint=ck,
                                amp_mode=am,
                                fp8=fp,
                            )
                        )
    if args.max_combos and len(combos) > args.max_combos:
        combos = combos[: args.max_combos]
    return combos


def _dpcs_config_for(toggle: Toggle, device: torch.device, base_cfg: Dict[str, object]) -> Dict[str, object]:
    cfg = dict(base_cfg)
    cfg["device_type"] = device.type
    cfg["enable_precision"] = 1 if toggle.amp_mode == "dynamic" else 0
    cfg["curv_period"] = max(0, int(toggle.curv_period))
    return cfg


def _disable_grad_signals(scheduler: DPCS) -> None:
    grads = getattr(scheduler, "_grads", None)
    if grads is None:
        return
    handles = getattr(grads, "_hooks", [])
    for handle in list(handles):
        try:
            handle.remove()
        except Exception:
            pass
    try:
        handles.clear()
    except Exception:
        pass
    scheduler._grads = None  # type: ignore[attr-defined]


def _sum_checkpoint_bytes(leaves: Iterable[object]) -> Tuple[int, int]:
    total_bytes = 0
    num_ckpt = 0
    for leaf in leaves:
        use_ckpt = bool(getattr(leaf, "use_ckpt", False))
        if not use_ckpt:
            continue
        num_ckpt += 1
        try:
            total_bytes += int(getattr(leaf, "activation_bytes", 0))
        except Exception:
            total_bytes += 0
    return total_bytes, num_ckpt


def _profile_context(cfg: ProfileCfg, combo_idx: int) -> "torch.profiler.profile | nullcontext":
    if not cfg.enabled:
        from contextlib import nullcontext

        return nullcontext(None)

    try:
        import torch.profiler as profiler
    except Exception:  # pragma: no cover - torch without profiler
        from contextlib import nullcontext

        return nullcontext(None)

    activities = [profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        try:
            activities.append(profiler.ProfilerActivity.CUDA)
        except Exception:
            pass

    trace_path = cfg.out_dir / f"combo_{combo_idx:03d}_trace.json"
    trace_handler = None
    if cfg.kind in {"trace", "both"}:

        def _handler(prof: "profiler.profile") -> None:
            try:
                prof.export_chrome_trace(str(trace_path))
            except Exception:
                pass

        trace_handler = _handler

    ctx = profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        on_trace_ready=trace_handler,
    )
    return ctx


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def run_single(
    toggle: Toggle,
    combo_idx: int,
    args: argparse.Namespace,
    device: torch.device,
    profile_cfg: ProfileCfg,
) -> Dict[str, object]:
    _seed_everything(args.seed + combo_idx)

    base_cfg = {
        "kappa_low": args.kappa_low,
        "kappa_high": args.kappa_high,
        "mode_patience": args.mode_patience,
        "log_every": max(1, args.steps + args.warmup),
    }
    cfg_kwargs = _dpcs_config_for(toggle, device, base_cfg)
    scheduler = DPCS(**cfg_kwargs)

    model = TinyTransformer(
        vocab=args.vocab,
        embed_dim=args.embed_dim,
        num_heads=args.heads,
        ff_dim=args.ff_dim,
        layers=args.layers,
    )
    model.to(device)
    model.train()

    model = scheduler.wrap(model)

    if toggle.signals == "off":
        _disable_grad_signals(scheduler)

    if toggle.checkpoint == "off":
        scheduler.enable_checkpointing(False)
    else:
        scheduler.enable_checkpointing(True)

    if toggle.amp_mode == "fixed":
        mode = args.fixed_precision
        try:
            scheduler.force_precision(mode)
        except Exception:
            scheduler.force_precision("fp32")
    else:
        scheduler.clear_precision_override()

    fp8_requested = toggle.fp8 == "on"

    opt = optim.AdamW(model.parameters(), lr=args.lr)

    scaler: Optional[torch.amp.GradScaler]
    if device.type == "cuda":
        try:
            scaler = torch.amp.GradScaler(enabled=True)
        except Exception:
            scaler = None
    else:
        scaler = None

    criterion = nn.CrossEntropyLoss()
    tokens_per_step = args.batch * args.seq

    steps = max(0, int(args.steps))
    warmup = max(0, int(args.warmup))
    total_steps = steps + warmup

    def _empty_metrics() -> Dict[str, Optional[float]]:
        return {
            "tokens_per_second": None,
            "step_latency_ms_mean": None,
            "step_latency_ms_p50": None,
            "step_latency_ms_p95": None,
            "amp_overflow_steps": None,
            "oom_count": None,
            "bytes_checkpointed": None,
            "checkpointed_modules": None,
            "measured_steps": None,
        }

    fp8_available = bool(getattr(scheduler, "_fp8_supported", False))

    toggles_dict = {
        "signals": toggle.signals,
        "curv_period": toggle.curv_period,
        "checkpoint": toggle.checkpoint,
        "amp_mode": toggle.amp_mode,
        "fp8": toggle.fp8,
    }

    base_record: Dict[str, object] = {
        "combo_index": combo_idx,
        "toggles": toggles_dict,
        "device": device.type,
        "seed": args.seed + combo_idx,
        "steps": steps,
        "warmup": warmup,
        "model": {
            "vocab": args.vocab,
            "embed_dim": args.embed_dim,
            "ff_dim": args.ff_dim,
            "layers": args.layers,
            "heads": args.heads,
            "seq": args.seq,
            "batch": args.batch,
        },
        "metrics": _empty_metrics(),
        "precision_counts": {},
        "fp8_available": fp8_available,
        "status": "ok",
        "reason": None,
        "profiler": {
            "table": None,
            "trace": None,
            "error": None,
        },
    }

    if fp8_requested and not fp8_available:
        base_record["status"] = "skipped"
        base_record["reason"] = "fp8_not_supported"
        return base_record

    latencies_ms: List[float] = []
    ckpt_bytes_sum = 0
    ckpt_modules_sum = 0
    precision_counts: Dict[str, int] = {}
    overflow_count = 0
    oom_count = 0
    measured_tokens = 0
    measured_time = 0.0

    profile_table: Optional[str] = None
    trace_file: Optional[str] = None
    profile_error: Optional[str] = None

    if profile_cfg.enabled:
        profile_cfg.out_dir.mkdir(parents=True, exist_ok=True)

    prof = None
    try:
        with _profile_context(profile_cfg, combo_idx) as prof:
            for step in range(total_steps):
                scheduler.start_step()
                opt.zero_grad(set_to_none=True)

                tokens = torch.randint(0, args.vocab, (args.batch, args.seq), device=device)
                targets = torch.randint(0, args.vocab, (args.batch,), device=device)

                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                t0 = time.perf_counter()

                try:
                    with scheduler.forward_context():
                        dev_type, amp_dtype, amp_enabled = scheduler.get_amp_config()
                        with torch.autocast(device_type=dev_type, dtype=amp_dtype, enabled=amp_enabled):
                            logits = model(tokens)
                            loss = criterion(logits, targets)

                    use_scaler = scaler is not None and scheduler.amp_uses_grad_scaler()
                    if use_scaler:
                        assert scaler is not None  # for type checkers
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    if toggle.signals == "on":
                        scheduler.collect_signals(loss, model)
                    else:
                        scheduler.collect_signals(None, None)

                    step_overflow = False
                    if use_scaler and scaler is not None:
                        scaler.step(opt)
                        monitor = scheduler.overflow_monitor(scaler)
                        with monitor:
                            scaler.update()
                        step_overflow = bool(monitor.last_overflow)
                    else:
                        opt.step()

                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower():
                        oom_count += 1
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                        scheduler.collect_signals(None, None)
                        scheduler.end_step(opt, scaler if scaler is not None else None)
                        continue
                    raise

                finally:
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)

                elapsed_ms = (time.perf_counter() - t0) * 1000.0

                scheduler.end_step(opt, scaler if scaler is not None else None)

                precision_counts[scheduler._amp_mode] = precision_counts.get(scheduler._amp_mode, 0) + 1

                if step >= warmup:
                    latencies_ms.append(elapsed_ms)
                    measured_tokens += tokens_per_step
                    measured_time += elapsed_ms / 1000.0
                    leaves = getattr(scheduler, "_leaves", [])
                    ckpt_bytes, ckpt_mods = _sum_checkpoint_bytes(leaves)
                    ckpt_bytes_sum += ckpt_bytes
                    ckpt_modules_sum += ckpt_mods
                    if use_scaler:
                        overflow_count += int(step_overflow)

                if prof is not None:
                    prof.step()

    except Exception as exc:
        profile_error = str(exc)

    else:
        if profile_cfg.enabled:
            try:
                import torch.profiler as profiler  # type: ignore

                if prof is not None and isinstance(prof, profiler.profile):  # type: ignore[arg-type]
                    if profile_cfg.kind in {"table", "both"}:
                        try:
                            profile_table = prof.key_averages().table(
                                sort_by="self_cpu_time_total", row_limit=15
                            )
                        except Exception:
                            profile_table = None
                    if profile_cfg.kind in {"trace", "both"}:
                        trace_candidate = profile_cfg.out_dir / f"combo_{combo_idx:03d}_trace.json"
                        if trace_candidate.exists():
                            trace_file = str(trace_candidate)
            except Exception as exc:  # pragma: no cover - profiler failures rare
                profile_error = str(exc)

    metrics = base_record["metrics"]
    assert isinstance(metrics, dict)

    if latencies_ms:
        measured_steps = len(latencies_ms)
        tokens_per_sec = (measured_tokens / measured_time) if measured_time > 0 else 0.0
        metrics.update(
            {
                "tokens_per_second": tokens_per_sec,
                "step_latency_ms_mean": sum(latencies_ms) / measured_steps,
                "step_latency_ms_p50": _quantile(latencies_ms, 0.5),
                "step_latency_ms_p95": _quantile(latencies_ms, 0.95),
                "amp_overflow_steps": overflow_count,
                "oom_count": oom_count,
                "bytes_checkpointed": ckpt_bytes_sum / measured_steps,
                "checkpointed_modules": ckpt_modules_sum / measured_steps,
                "measured_steps": measured_steps,
            }
        )
    else:
        metrics.update(
            {
                "tokens_per_second": 0.0,
                "step_latency_ms_mean": 0.0,
                "step_latency_ms_p50": 0.0,
                "step_latency_ms_p95": 0.0,
                "amp_overflow_steps": overflow_count,
                "oom_count": oom_count,
                "bytes_checkpointed": 0.0,
                "checkpointed_modules": 0.0,
                "measured_steps": 0,
            }
        )

    base_record["precision_counts"] = precision_counts
    base_record["profiler"] = {
        "table": profile_table,
        "trace": trace_file,
        "error": profile_error,
    }

    return base_record


def _write_jsonl(path: Path, record: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        json.dump(record, fh)
        fh.write("\n")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DPCS micro-benchmark harness")
    parser.add_argument("--device", default="auto", help="Device to run on (cuda|cpu|auto)")
    parser.add_argument("--steps", type=int, default=20, help="Timed training steps per combo")
    parser.add_argument("--warmup", type=int, default=4, help="Warm-up steps excluded from metrics")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--seq", type=int, default=128, help="Sequence length")
    parser.add_argument("--vocab", type=int, default=256, help="Vocabulary size")
    parser.add_argument("--embed-dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--ff-dim", type=int, default=256, help="Feed-forward hidden dimension")
    parser.add_argument("--layers", type=int, default=2, help="Number of Transformer blocks")
    parser.add_argument("--heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate for AdamW")
    parser.add_argument("--seed", type=int, default=0, help="Random seed base")
    parser.add_argument("--signals", nargs="+", choices=["on", "off"], help="Signal toggle list")
    parser.add_argument("--curv-periods", nargs="+", type=int, help="Curvature probe cadence values")
    parser.add_argument(
        "--checkpoint-modes",
        nargs="+",
        choices=["off", "auto"],
        help="Activation checkpointing modes",
    )
    parser.add_argument(
        "--amp-modes",
        nargs="+",
        choices=["fixed", "dynamic"],
        help="AMP precision control modes",
    )
    parser.add_argument(
        "--fp8",
        nargs="+",
        choices=["off", "on"],
        help="TransformerEngine FP8 toggle list",
    )
    parser.add_argument(
        "--fixed-precision",
        default="auto",
        help="Precision to enforce when amp-mode=fixed (fp32|bf16|fp16|auto)",
    )
    parser.add_argument(
        "--max-combos",
        type=int,
        default=0,
        help="Optional limit on combinations to execute (0 = no limit)",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "bench" / "bench_dpcs.jsonl"),
        help="Path to JSONL output file",
    )
    parser.add_argument(
        "--profile",
        default="none",
        choices=["none", "table", "trace", "both"],
        help="Enable torch.profiler capture",
    )
    parser.add_argument(
        "--profile-dir",
        default=str(ROOT / "bench" / "profiler"),
        help="Directory for profiler outputs",
    )
    parser.add_argument(
        "--kappa-low",
        type=float,
        default=1e-3,
        help="Lower curvature band override for reproducibility",
    )
    parser.add_argument(
        "--kappa-high",
        type=float,
        default=1e-1,
        help="Upper curvature band override for reproducibility",
    )
    parser.add_argument(
        "--mode-patience",
        type=int,
        default=8,
        help="Precision policy patience (affects overflow windows)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    device = _auto_device(args.device)
    if str(args.fixed_precision).lower() == "auto":
        fixed = "fp32"
        if device.type == "cuda" and torch.cuda.is_available():
            try:
                if torch.cuda.is_bf16_supported():
                    fixed = "bf16"
                else:
                    fixed = "fp16"
            except Exception:
                fixed = "fp16"
        args.fixed_precision = fixed
    toggles = _expand_toggles(args)
    profile_cfg = ProfileCfg(
        enabled=args.profile != "none",
        kind=args.profile,
        out_dir=Path(args.profile_dir),
    )

    output_path = Path(args.output)
    if output_path.exists():
        output_path.unlink()

    for idx, toggle in enumerate(toggles):
        record = run_single(toggle, idx, args, device, profile_cfg)
        _write_jsonl(output_path, record)
        status = str(record.get("status", "ok"))
        print(
            f"[{idx + 1}/{len(toggles)}] {status}: signals={toggle.signals} curv={toggle.curv_period} "
            f"ckpt={toggle.checkpoint} amp={toggle.amp_mode} fp8={toggle.fp8}",
            flush=True,
        )

    print(f"Results written to {output_path}")


if __name__ == "__main__":
    main()

