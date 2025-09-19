import os
import sys

import pytest
import torch
import torch.nn as nn


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from dpcs import DPCS, PrecisionCfg  # noqa: E402


def _tiny_model() -> nn.Module:
    return nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 64)).cuda().train()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for AMP tests")
def test_scheduler_demotes_after_patience():
    torch.manual_seed(0)
    model = _tiny_model()
    dpcs = DPCS(device_type="cuda", enable_precision=1, precision_cfg=dict(patience=2))
    model = dpcs.wrap(model)
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)

    for _ in range(2):
        dpcs.start_step()
        opt.zero_grad(set_to_none=True)
        x = torch.randn(4, 64, device="cuda")
        y = model(x)
        loss = (y.float() ** 2).mean()
        loss.backward()
        dpcs.collect_signals(loss, model)
        opt.step()
        dpcs.end_step(opt, None)

    assert dpcs._amp_mode == dpcs._prec_pol.low_mode


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for AMP tests")
def test_scheduler_overflow_triggers_fp32_cooldown():
    torch.manual_seed(1)
    model = _tiny_model()
    prec_cfg = PrecisionCfg(prefer_bf16=False, patience=2)
    dpcs = DPCS(device_type="cuda", enable_precision=1, precision_cfg=prec_cfg)
    model = dpcs.wrap(model)
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler(device="cuda")

    for _ in range(prec_cfg.patience):
        dpcs.start_step()
        opt.zero_grad(set_to_none=True)
        x = torch.randn(8, 64, device="cuda")
        y = model(x)
        loss = (y.float() ** 2).mean()
        scaler.scale(loss).backward()
        dpcs.collect_signals(loss, model)
        scaler.step(opt)
        with dpcs.overflow_monitor(scaler):
            scaler.update()
        dpcs.end_step(opt, scaler)

    assert dpcs._amp_mode == "fp16"

    dpcs.start_step()
    opt.zero_grad(set_to_none=True)
    x = torch.randn(8, 64, device="cuda")
    y = model(x)
    loss = (y.float() ** 2).mean()
    scaler.scale(loss).backward()
    first_param = next(model.parameters())
    assert first_param.grad is not None
    first_param.grad.view(-1)[0] = float("nan")
    dpcs.collect_signals(loss, model)
    scaler.step(opt)
    with dpcs.overflow_monitor(scaler):
        scaler.update()
    dpcs.end_step(opt, scaler)

    assert dpcs._amp_mode == "fp32"
    assert dpcs._prec_pol.cooldown == 2

    while dpcs._prec_pol.cooldown > 0:
        dpcs.start_step()
        opt.zero_grad(set_to_none=True)
        x = torch.randn(8, 64, device="cuda")
        y = model(x)
        loss = (y.float() ** 2).mean()
        scaler.scale(loss).backward()
        dpcs.collect_signals(loss, model)
        scaler.step(opt)
        with dpcs.overflow_monitor(scaler):
            scaler.update()
        dpcs.end_step(opt, scaler)
        assert dpcs._amp_mode == "fp32"

    for _ in range(prec_cfg.patience):
        dpcs.start_step()
        opt.zero_grad(set_to_none=True)
        x = torch.randn(8, 64, device="cuda")
        y = model(x)
        loss = (y.float() ** 2).mean()
        scaler.scale(loss).backward()
        dpcs.collect_signals(loss, model)
        scaler.step(opt)
        with dpcs.overflow_monitor(scaler):
            scaler.update()
        dpcs.end_step(opt, scaler)

    assert dpcs._amp_mode == "fp16"
