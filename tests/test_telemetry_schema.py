import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dpcs.runtime import JsonlLogger, TELEMETRY_SCHEMA


REQUIRED_FIELDS = {
    "step",
    "broadcast_step",
    "headroom_frac",
    "amp_mode",
    "overflow",
    "overflow_flag",
    "cooldown_remaining",
    "num_checkpointed",
    "ckpt_on",
    "num_leaves",
    "step_peak_bytes",
    "peak_alloc_bytes",
    "allocated",
    "reserved",
    "active",
    "fragmentation_hint",
    "device_free",
    "device_total",
    "precision_mix",
}


def test_required_fields_present():
    required = {name for name, spec in TELEMETRY_SCHEMA.items() if spec.get("required")}
    missing = REQUIRED_FIELDS - required
    assert not missing, f"schema missing required fields: {sorted(missing)}"


def _base_record():
    return {
        "step": 1,
        "broadcast_step": 1,
        "headroom_frac": 0.75,
        "amp_mode": "fp32",
        "overflow": False,
        "overflow_flag": False,
        "cooldown_remaining": 0,
        "num_checkpointed": 0,
        "ckpt_on": False,
        "num_leaves": 1,
        "step_peak_bytes": 1024,
        "peak_alloc_bytes": 1024,
        "allocated": 1024,
        "reserved": 2048,
        "active": 1024,
        "fragmentation_hint": 0.0,
        "device_free": 512,
        "device_total": 4096,
        "precision_mix": {"fp32": 1},
        "free_bytes": 512,
        "total_bytes": 4096,
        "grad_var_avg": None,
        "curv_avg": None,
        "ckpt_ids": [],
        "ckpt_modules": [],
        "graph_breaks_total": 0,
        "top_break_reasons": [],
    }


def test_jsonl_logger_writes_valid_json(tmp_path: Path):
    record = _base_record()
    record.update({
        "ckpt_ids": (1, 2),
        "ckpt_modules": ("layer1", "layer2"),
        "top_break_reasons": (("reason", 3),),
    })
    logger = JsonlLogger(tmp_path / "telemetry.jsonl", flush_every=1)
    logger.log(record)
    logger.close()

    data = (tmp_path / "telemetry.jsonl").read_text().strip().splitlines()
    assert len(data) == 1
    payload = json.loads(data[0])

    for key, spec in TELEMETRY_SCHEMA.items():
        if spec.get("required"):
            assert key in payload
        if key in payload:
            value = payload[key]
            if value is None:
                assert spec.get("allow_none"), f"{key} should not be None"

    assert payload["ckpt_ids"] == [1, 2]
    assert payload["ckpt_modules"] == ["layer1", "layer2"]
    assert payload["top_break_reasons"] == [{"reason": "reason", "count": 3}]


def test_jsonl_logger_rejects_invalid_records(tmp_path: Path):
    logger = JsonlLogger(tmp_path / "telemetry.jsonl", flush_every=1)

    bad_missing = _base_record()
    bad_missing.pop("step")
    with pytest.raises(ValueError):
        logger.log(bad_missing)

    bad_type = _base_record()
    bad_type["allocated"] = "lots"
    with pytest.raises(TypeError):
        logger.log(bad_type)

    logger.close()


def test_jsonl_logger_allows_absent_device_metrics(tmp_path: Path):
    record = _base_record()
    record.update({
        "device_free": None,
        "device_total": None,
        "headroom_frac": None,
    })

    logger = JsonlLogger(tmp_path / "telemetry.jsonl", flush_every=1)
    logger.log(record)
    logger.close()

    payload = json.loads((tmp_path / "telemetry.jsonl").read_text().strip())
    assert payload["device_free"] is None
    assert payload["device_total"] is None
    assert payload["headroom_frac"] is None
