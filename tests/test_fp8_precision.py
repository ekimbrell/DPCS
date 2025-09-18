import os
import sys

import pytest
import torch
import torch.nn as nn

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from dpcs import DPCS  # noqa: E402


pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for FP8 scheduler path"),
]


def _patch_headroom(monkeypatch, value_container):
    def _fake_headroom():
        return value_container["value"]

    monkeypatch.setattr("dpcs.scheduler.headroom_frac", _fake_headroom)


def _patch_mem_get_info(monkeypatch):
    def _fake_mem_info():
        return (1 << 29, 1 << 30)

    monkeypatch.setattr("dpcs.scheduler.mem_get_info", _fake_mem_info)


def test_dpcs_fp8_mode_demote_and_promote(monkeypatch):
    pytest.importorskip("transformer_engine.pytorch", reason="TransformerEngine required for FP8 tests")

    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(256, 256),
        nn.GELU(),
        nn.Linear(256, 256),
    ).cuda().train()

    dpcs = DPCS(
        device_type="cuda",
        enable_precision=1,
        epsilon_g_low=0.0,
        epsilon_g_high=1.0,
        kappa_low=0.0,
        kappa_high=1.0,
        low_headroom_frac=0.05,
        hi_headroom_frac=0.4,
    )

    model = dpcs.wrap(model)
    if not dpcs._fp8_supported or not dpcs._fp8_wrappers:
        pytest.skip("FP8 replacements not available on this platform")

    opt = torch.optim.SGD(model.parameters(), lr=1e-3)

    headroom = {"value": 0.01}
    _patch_headroom(monkeypatch, headroom)
    _patch_mem_get_info(monkeypatch)

    def run_step() -> bool:
        dpcs.start_step()
        fp8_active = any(w._fp8_active for w in dpcs._fp8_wrappers)
        opt.zero_grad(set_to_none=True)
        x = torch.randn(8, 256, device="cuda")
        y = model(x)
        loss = (y.float() ** 2).mean()
        loss.backward()
        dpcs.collect_signals(loss, model)
        opt.step()
        dpcs.end_step(opt, None)
        return fp8_active

    # Step 1: low headroom requests FP8 demotion
    headroom["value"] = 0.01
    active_step1 = run_step()
    assert not active_step1
    assert dpcs._amp_mode == "fp8"

    # Step 2: start in FP8, then request promotion with high headroom
    headroom["value"] = 0.8
    active_step2 = run_step()
    assert active_step2
    assert dpcs._amp_mode in {"bf16", "fp16", "fp32"}

    # Step 3: remain at high headroom; FP8 stays disabled at start
    headroom["value"] = 0.8
    active_step3 = run_step()
    assert not active_step3
    assert dpcs._amp_mode in {"bf16", "fp16", "fp32"}
