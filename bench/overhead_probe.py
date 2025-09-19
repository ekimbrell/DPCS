"""Micro benchmark for telemetry hook overhead.

This script runs two tiny models (Transformer-like and ResNet-like) in three
configurations to measure the overhead of activation and timing hooks. Each
configuration is executed for a configurable number of steps (default 200):

(A) Baseline - no hooks.
(B) Activation hooks only.
(C) Activation + timing hooks.

For every configuration and model we record the exponential moving average
(EMA) of the step time (in milliseconds) and the maximum per-step peak memory
reported by :func:`dpcs.runtime.get_step_peak`. If activation hooks add more
than 2% overhead or activation+timing hooks add more than 3% relative to the
baseline, the script exits with a non-zero status. All deltas and peak usage
are printed for CI visibility.
"""
from __future__ import annotations

import argparse
import math
import re
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if SRC_DIR.exists():
    src_str = str(SRC_DIR)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)

from dpcs.config import TelemetryCfg
from dpcs.runtime import get_step_peak, reset_step_peak
from dpcs.signals import ActivationBytesEMA, ForwardTimer


# ------------------------------ EMA helper ---------------------------------


class ScalarEMA:
    """Scalar EMA helper."""

    __slots__ = ("beta", "value", "initialized")

    def __init__(self, beta: float = 0.9) -> None:
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


# ----------------------------- Tiny models ---------------------------------


class TransformerBlock(nn.Module):
    """Minimal Transformer block with explicit attn/ffn submodules."""

    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.0, batch_first=True)
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
    """Tiny Transformer used for the micro benchmark."""

    def __init__(self, vocab: int = 128, embed_dim: int = 64, seq_len: int = 32, layers: int = 2) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, embed_dim)
        self.blocks = nn.Sequential(
            OrderedDict(
                [(f"block{i}", TransformerBlock(embed_dim, num_heads=4, ff_dim=128)) for i in range(layers)]
            )
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab)
        self.seq_len = seq_len

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens)
        x = self.blocks(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.head(x)


class ResidualBlock(nn.Module):
    """Small residual block with an optional downsample."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                OrderedDict(
                    [
                        ("conv", nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False)),
                        ("bn", nn.BatchNorm2d(out_ch)),
                    ]
                )
            )
        else:
            self.downsample = nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out + identity)
        return out


class TinyResNet(nn.Module):
    """Tiny ResNet-style network used for the micro benchmark."""

    def __init__(self, in_ch: int = 3, classes: int = 10) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            OrderedDict(
                [
                    ("conv", nn.Conv2d(in_ch, 32, kernel_size=3, stride=1, padding=1, bias=False)),
                    ("bn", nn.BatchNorm2d(32)),
                    ("relu", nn.ReLU(inplace=True)),
                ]
            )
        )
        self.blocks = nn.Sequential(
            OrderedDict(
                [
                    ("block0", ResidualBlock(32, 32, stride=1)),
                    ("block1", ResidualBlock(32, 32, stride=1)),
                    ("block2", ResidualBlock(32, 64, stride=2)),
                    ("block3", ResidualBlock(64, 64, stride=1)),
                ]
            )
        )
        self.head = nn.Sequential(
            OrderedDict(
                [
                    ("pool", nn.AdaptiveAvgPool2d(1)),
                    ("flatten", nn.Flatten()),
                    ("fc", nn.Linear(64, classes)),
                ]
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x


# ------------------------------ Hook setup ---------------------------------


HOOK_PATTERN = re.compile(r"(attn|ffn|block)")


def attach_activation_hooks(model: nn.Module) -> Optional[ActivationBytesEMA]:
    named = list(model.named_modules())
    cfg = TelemetryCfg(sample_every=1, sampled_modules_regex=HOOK_PATTERN.pattern)
    act = ActivationBytesEMA(named, cfg, beta=0.9)
    act.attach()
    return act if act.enabled else None


def attach_timing_hooks(model: nn.Module) -> Tuple[List[ForwardTimer], List[torch.utils.hooks.RemovableHandle]]:
    timers: List[ForwardTimer] = []
    handles: List[torch.utils.hooks.RemovableHandle] = []
    cfg = TelemetryCfg(enable_timing=True)

    def make_pre(timer: ForwardTimer):
        def _pre(_module: nn.Module, _inputs):
            timer.start()

        return _pre

    def make_post(timer: ForwardTimer):
        def _post(_module: nn.Module, _inputs, _output):
            timer.stop_update()

        return _post

    for name, module in model.named_modules():
        if name and HOOK_PATTERN.search(name):
            timer = ForwardTimer(beta=0.9, telemetry=cfg)
            timers.append(timer)
            handles.append(module.register_forward_pre_hook(make_pre(timer)))
            handles.append(module.register_forward_hook(make_post(timer)))
    return timers, handles


# ------------------------------ Benchmark ----------------------------------


@dataclass
class BenchResult:
    ema_ms: float
    peak_bytes: int
    compiled: bool = False


def synthetic_transformer_batch(batch_size: int, seq_len: int, vocab: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    tokens = torch.randint(0, vocab, (batch_size, seq_len), device=device)
    targets = torch.randint(0, vocab, (batch_size,), device=device)
    return tokens, targets


def synthetic_resnet_batch(batch_size: int, image_size: int, classes: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    images = torch.randn(batch_size, 3, image_size, image_size, device=device)
    targets = torch.randint(0, classes, (batch_size,), device=device)
    return images, targets


def run_model(
    name: str,
    model_ctor: Callable[[], nn.Module],
    data_fn: Callable[[torch.device], Tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    steps: int,
    enable_activation: bool,
    enable_timing: bool,
    compile_mode: bool = False,
) -> BenchResult:
    compile_requested = compile_mode
    use_activation = enable_activation
    use_timing = enable_timing

    if compile_requested and (use_activation or use_timing):
        print(
            f"[{name}] compile_mode=True disables activation/timing hooks to avoid graph breaks.",
            file=sys.stderr,
        )
        use_activation = False
        use_timing = False

    model = model_ctor().to(device)
    model.train(True)

    compile_fn = getattr(torch, "compile", None)
    compiled = False
    if compile_requested:
        if compile_fn is None:
            print(f"[{name}] WARNING: torch.compile is unavailable; running in eager mode.", file=sys.stderr)
        else:
            try:
                model = compile_fn(model, backend="inductor")
                compiled = True
            except Exception as exc:  # pragma: no cover - defensive, compile path is best effort
                print(
                    f"[{name}] WARNING: torch.compile(model) failed ({exc}). Falling back to eager mode.",
                    file=sys.stderr,
                )

    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    activation = attach_activation_hooks(model) if use_activation else None
    timers: List[ForwardTimer] = []
    timing_handles: List[torch.utils.hooks.RemovableHandle] = []
    if use_timing:
        timers, timing_handles = attach_timing_hooks(model)

    ema = ScalarEMA(beta=0.9)
    peak_max = 0

    inputs, targets = data_fn(device)

    warmup_steps = 1 if compiled else 0
    total_steps = steps + warmup_steps

    for raw_step in range(total_steps):
        logical_step = raw_step - warmup_steps
        if activation is not None and logical_step >= 0:
            activation.set_step(logical_step)
        if device.type == "cuda":
            torch.cuda.synchronize()
        reset_step_peak(device)
        t0 = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if logical_step >= 0:
            ema.update(elapsed_ms)
            peak = get_step_peak(device)
            peak_max = max(peak_max, peak)

    if activation is not None:
        activation.detach()
    for handle in timing_handles:
        try:
            handle.remove()
        except Exception:
            pass

    timers.clear()

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return BenchResult(ema_ms=ema.value, peak_bytes=peak_max, compiled=compiled)


# ------------------------------ CLI runner ---------------------------------


def format_bytes(num_bytes: int) -> str:
    if num_bytes <= 0:
        return "0 B"
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    idx = min(len(units) - 1, int(math.log(num_bytes, 1024)))
    scaled = num_bytes / (1024 ** idx)
    return f"{scaled:.2f} {units[idx]}"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=200, help="Steps per configuration")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--compile-mode",
        action="store_true",
        help="Run an additional torch.compile (inductor) comparison for the baseline mode.",
    )
    args = ap.parse_args(argv)

    device = torch.device(args.device)
    torch.manual_seed(42)

    steps = max(1, int(args.steps))

    def transformer_ctor() -> nn.Module:
        return TinyTransformer()

    def resnet_ctor() -> nn.Module:
        return TinyResNet()

    vocab = 128
    seq_len = 32
    t_batch = 16
    r_batch = 16
    image_size = 32
    num_classes = 10

    def transformer_data(dev: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        return synthetic_transformer_batch(t_batch, seq_len, vocab, dev)

    def resnet_data(dev: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        return synthetic_resnet_batch(r_batch, image_size, num_classes, dev)

    benches = OrderedDict(
        [
            ("transformer", (transformer_ctor, transformer_data)),
            ("resnet", (resnet_ctor, resnet_data)),
        ]
    )

    modes = OrderedDict(
        [
            ("A", {"activation": False, "timing": False, "label": "no hooks"}),
            ("B", {"activation": True, "timing": False, "label": "activation hooks"}),
            ("C", {"activation": True, "timing": True, "label": "activation + timing"}),
        ]
    )

    results: Dict[str, Dict[str, BenchResult]] = {}

    for mode, cfg in modes.items():
        results[mode] = {}
        for bench_name, (ctor, data_fn) in benches.items():
            result = run_model(
                bench_name,
                ctor,
                data_fn,
                device=device,
                steps=steps,
                enable_activation=cfg["activation"],
                enable_timing=cfg["timing"],
            )
            results[mode][bench_name] = result
            print(
                f"Mode {mode} ({cfg['label']}), {bench_name}: "
                f"step_ema={result.ema_ms:.3f} ms, peak={format_bytes(result.peak_bytes)}"
            )

    compile_speedups: Dict[str, float] = {}
    if args.compile_mode:
        expected_speedup = 1.0
        for bench_name, (ctor, data_fn) in benches.items():
            eager_result = results.get("A", {}).get(bench_name)
            if eager_result is None:
                continue
            compile_result = run_model(
                bench_name,
                ctor,
                data_fn,
                device=device,
                steps=steps,
                enable_activation=False,
                enable_timing=False,
                compile_mode=True,
            )
            if not compile_result.compiled:
                print(
                    f"[{bench_name}] WARNING: torch.compile run was skipped; speedup comparison unavailable.",
                    file=sys.stderr,
                )
                continue
            compile_speedups[bench_name] = (
                0.0
                if compile_result.ema_ms <= 0
                else eager_result.ema_ms / compile_result.ema_ms
            )
            speedup = compile_speedups[bench_name]
            print(
                f"[compile] {bench_name}: eager={eager_result.ema_ms:.3f} ms -> "
                f"compile={compile_result.ema_ms:.3f} ms ({speedup:.2f}x)",
            )
            if speedup + 1e-9 < expected_speedup:
                print(
                    f"WARNING: {bench_name} torch.compile speedup {speedup:.2f}x < expected {expected_speedup:.2f}x.",
                    file=sys.stderr,
                )
        if compile_speedups:
            mean_speedup = sum(compile_speedups.values()) / len(compile_speedups)
            print(
                f"Mean torch.compile speedup across benches: {mean_speedup:.2f}x",
            )

    def mean_step_ema(mode_results: Dict[str, BenchResult]) -> float:
        if not mode_results:
            return 0.0
        return sum(res.ema_ms for res in mode_results.values()) / len(mode_results)

    baseline = mean_step_ema(results["A"])
    exit_code = 0

    for mode in ("B", "C"):
        ema = mean_step_ema(results[mode])
        delta = 0.0 if baseline <= 0 else (ema / baseline - 1.0) * 100.0
        budget = 2.0 if mode == "B" else 3.0
        status = "OK"
        if delta > budget + 1e-6:
            status = "FAIL"
            exit_code = 1
        print(f"Mode {mode} delta vs A: {delta:+.2f}% step time (budget {budget:.1f}%) -> {status}")

    peak_info = {
        mode: max(res.peak_bytes for res in mode_results.values()) for mode, mode_results in results.items()
    }
    for mode, peak in peak_info.items():
        print(f"Mode {mode} peak across benches: {format_bytes(peak)}")

    if exit_code != 0:
        print(
            "Overhead guardrail violated: activation hooks must stay within 2% and activation+timing within 3%.",
            file=sys.stderr,
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
