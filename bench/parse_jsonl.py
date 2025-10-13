#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dpcs.runtime import TELEMETRY_SCHEMA, validate_telemetry_record  # noqa: E402

SUMMARY_FIELDS = [
    field
    for field in (
        "headroom_frac",
        "allocated",
        "reserved",
        "active",
        "fragmentation_hint",
        "device_free",
        "device_total",
        "step_peak_bytes",
        "peak_alloc_bytes",
        "num_checkpointed",
    )
    if field in TELEMETRY_SCHEMA
]


def _load_records(path: Path) -> List[Mapping[str, Any]]:
    records: List[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for idx, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"Skipping line {idx}: invalid JSON ({exc})", file=sys.stderr)
                continue
            try:
                validated = validate_telemetry_record(data)
            except Exception as exc:  # pragma: no cover - defensive
                print(f"Skipping line {idx}: {exc}", file=sys.stderr)
                continue
            records.append(validated)
    return records


def _median_last(values: Iterable[Optional[float]], window: int) -> Optional[float]:
    recent: List[float] = [float(v) for v in values if v is not None]
    if not recent:
        return None
    window = max(1, window)
    subset = recent[-window:]
    try:
        return float(statistics.median(subset))
    except statistics.StatisticsError:  # pragma: no cover - defensive
        return None


def _format_value(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int,)):
        return f"{value:,}"
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{as_float:.4f}"


def _summary_line(field: str, last: Any, median: Optional[float], window: int) -> str:
    median_str = "NA" if median is None else f"{median:.4f}"
    return f"{field:>20}: last={_format_value(last):>12} | median[{window}]={median_str:>12}"


def summarize(records: List[Mapping[str, Any]], window: int) -> None:
    total = len(records)
    print(f"Loaded {total} records")
    if not records:
        return

    latest = records[-1]
    step = latest.get("step")
    broadcast = latest.get("broadcast_step")
    amp_mode = latest.get("amp_mode")
    overflow = latest.get("overflow")
    ckpt_on = latest.get("ckpt_on")
    precision_mix = latest.get("precision_mix", {})

    print(f"Latest step: {step} (broadcast {broadcast})")
    print(f"AMP mode: {amp_mode} | overflow: {_format_value(overflow)} | ckpt_on: {_format_value(ckpt_on)}")
    if isinstance(precision_mix, Mapping) and precision_mix:
        mix = ", ".join(f"{k}:{v}" for k, v in precision_mix.items())
        print(f"Precision mix: {mix}")

    for field in SUMMARY_FIELDS:
        values = [record.get(field) for record in records]
        numeric_values = [float(v) for v in values if isinstance(v, (int, float))]
        median = _median_last(numeric_values, window) if numeric_values else None
        last = values[-1] if values else None
        if median is None and last is None:
            continue
        print(_summary_line(field, last, median, window))

    device_total = latest.get("device_total")
    device_free = latest.get("device_free")
    if isinstance(device_total, (int, float)) and isinstance(device_free, (int, float)) and device_total:
        util = 1.0 - (float(device_free) / float(device_total))
        print(f"Device util: {util:.4f}")


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize DPCS telemetry JSONL logs")
    parser.add_argument("path", type=Path, help="Path to the telemetry JSONL log")
    parser.add_argument(
        "--window",
        type=int,
        default=50,
        help="Rolling window (number of records) for medians",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    if args.window <= 0:
        raise SystemExit("--window must be > 0")
    if not args.path.exists():
        raise SystemExit(f"No such file: {args.path}")
    records = _load_records(args.path)
    summarize(records, args.window)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
