"""Smoke-test the DPCS micro-benchmark harness."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def test_bench_smoke(tmp_path: Path) -> None:
    """Ensure the benchmark runs and emits the expected metrics."""

    output = tmp_path / "bench_metrics.jsonl"
    cmd = [
        sys.executable,
        "-m",
        "bench.bench_dpcs",
        "--device",
        "cpu",
        "--steps",
        "2",
        "--warmup",
        "0",
        "--batch",
        "2",
        "--seq",
        "8",
        "--signals",
        "on",
        "--curv-periods",
        "0",
        "--checkpoint-modes",
        "off",
        "--amp-modes",
        "fixed",
        "--fp8",
        "off",
        "--max-combos",
        "1",
        "--output",
        str(output),
    ]

    subprocess.run(cmd, check=True, cwd=_repo_root())

    lines = output.read_text().strip().splitlines()
    assert len(lines) == 1, "expected exactly one benchmark record"
    record = json.loads(lines[0])
    metrics = record.get("metrics", {})

    for key in [
        "tokens_per_second",
        "step_latency_ms_mean",
        "step_latency_ms_p50",
        "step_latency_ms_p95",
        "amp_overflow_steps",
        "oom_count",
        "bytes_checkpointed",
        "checkpointed_modules",
        "measured_steps",
    ]:
        assert key in metrics, f"missing metric: {key}"
        assert metrics[key] is not None

    assert record.get("toggles", {}).get("signals") == "on"
    assert record.get("device") == "cpu"
    assert record.get("status") == "ok"
    assert record.get("profiler") is not None
