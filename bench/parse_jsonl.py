#!/usr/bin/env python3
"""Parse DPCS telemetry JSONL logs and report simple summaries."""

from __future__ import annotations

import argparse
import json
from collections import deque
from statistics import median
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Sequence

NUMERIC_FIELDS = [
    "headroom_frac",
    "allocated",
    "reserved",
    "active",
    "fragmentation_hint",
    "device_free",
    "device_total",
    "step_peak_bytes",
    "peak_alloc_bytes",
]


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, Mapping):
                records.append(dict(parsed))
    return records


def rolling_median(values: Iterable[float], window: int) -> Iterator[float]:
    window = max(1, int(window))
    buf: deque[float] = deque(maxlen=window)
    for value in values:
        buf.append(float(value))
        yield median(buf)


def numeric_series(records: Sequence[Mapping[str, Any]], field: str) -> List[float]:
    series: List[float] = []
    for record in records:
        value = record.get(field)
        if isinstance(value, (int, float)):
            series.append(float(value))
    return series


def summarize(records: Sequence[Mapping[str, Any]], window: int) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "count": len(records),
    }
    if records:
        summary["last_step"] = records[-1].get("step")
    for field in NUMERIC_FIELDS:
        values = numeric_series(records, field)
        if not values:
            continue
        medians = list(rolling_median(values, window))
        summary[field] = {
            "latest": values[-1],
            "median": medians[-1],
        }
    return summary


def format_value(value: float) -> str:
    if abs(value) >= 1:
        return f"{value:,.2f}"
    return f"{value:.4f}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Path to telemetry JSONL file")
    parser.add_argument("--window", type=int, default=5, help="Rolling median window size")
    args = parser.parse_args(argv)

    records = load_jsonl(args.path)
    summary = summarize(records, args.window)

    print(f"Loaded {summary.get('count', 0)} records from {args.path}")
    last_step = summary.get("last_step")
    if last_step is not None:
        print(f"Last step: {last_step}")
    for field in NUMERIC_FIELDS:
        info = summary.get(field)
        if not info:
            continue
        latest = format_value(float(info["latest"]))
        median_val = format_value(float(info["median"]))
        print(f"{field:>20}: last={latest} median={median_val}")
    return 0


if __name__ == "__main__":  # pragma: no cover - manual utility
    raise SystemExit(main())
