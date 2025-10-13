from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dpcs.runtime import (
    JsonlLogger,
    TELEMETRY_SCHEMA,
    validate_telemetry_record,
)


def _example_record() -> dict:
    return {
        "step": 12,
        "broadcast_step": 3,
        "headroom_frac": 0.5,
        "amp_mode": "bf16",
        "overflow": False,
        "overflow_flag": False,
        "cooldown_remaining": 0,
        "num_checkpointed": 2,
        "ckpt_on": True,
        "num_leaves": 16,
        "step_peak_bytes": 4096,
        "peak_alloc_bytes": 4096,
        "allocated": 2048,
        "reserved": 4096,
        "active": 3000,
        "fragmentation_hint": 0.25,
        "device_free": 8192,
        "device_total": 16384,
        "precision_mix": {"bf16": 16},
        "grad_var_avg": 0.01,
        "curv_avg": None,
        "ckpt_ids": [1, 5],
        "ckpt_modules": ["layer1", "layer5"],
        "graph_breaks_total": 0,
        "top_break_reasons": [{"reason": "guard", "count": 1}],
    }


def test_schema_includes_memory_fields() -> None:
    required = {name for name, spec in TELEMETRY_SCHEMA.items() if spec.get("required")}
    assert {"allocated", "reserved", "active", "fragmentation_hint", "device_free", "device_total", "headroom_frac"}.issubset(
        required
    )


def test_jsonl_logger_produces_valid_records(tmp_path: Path) -> None:
    path = tmp_path / "telemetry.jsonl"
    logger = JsonlLogger(str(path), flush_every=1)
    record = _example_record()
    logger.log(record)
    logger.flush()
    logger.close()

    content = path.read_text(encoding="utf-8").strip()
    assert content
    parsed = json.loads(content)
    expected = validate_telemetry_record(record)
    assert parsed == expected
    assert set(parsed) == set(TELEMETRY_SCHEMA)
    assert validate_telemetry_record(parsed) == parsed
