import os, types, torch, torch.nn as nn, pytest
from dpcs import DPCS

CUDA = torch.cuda.is_available()
DEV = "cuda" if CUDA else "cpu"

@pytest.mark.skipif(not CUDA, reason="CUDA-only test")
def test_fp32_override_under_autocast():
    m = nn.Linear(64,64).to(DEV)
    d = DPCS(device_type=DEV)
    m = d.wrap(m, allow_fp8=False)
    # force FP32 mode for the only leaf module
    st = next(iter(d._registry.values()))
    st.mode = "fp32"
    x = torch.randn(8,64, device=DEV)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        y = m(x)
    assert y.dtype == torch.float32  # local disable autocast worked

@pytest.mark.skipif(not CUDA, reason="CUDA-only test")
def test_ckpt_gate_hysteresis(monkeypatch):
    d = DPCS(device_type="cuda", ckpt_low=0.9, ckpt_high=0.95, ckpt_need=2)
    # simulate low headroom twice → turns on
    calls = [ (100,1000), (80,1000), (80,1000), (980,1000) ]  # free,total
    it = iter(calls)
    monkeypatch.setattr(torch.cuda.memory, "mem_get_info", lambda : next(it))
    # wrap a trivial model and tick end_step to update gate
    m = nn.Linear(8,8).cuda(); d.wrap(m)
    for _ in range(3): d.end_step(opt := torch.optim.SGD(m.parameters(), 0.1))
    assert d.is_checkpointing() is True
    # next call returns high headroom -> turns off
    d.end_step(opt)
    assert d.is_checkpointing() is False

@pytest.mark.skipif(not CUDA, reason="CUDA-only test")
def test_overflow_forces_fp32():
    torch.manual_seed(0)
    m = nn.Linear(256,256).to(DEV)
    d = DPCS(device_type=DEV)
    m = d.wrap(m)
    opt = torch.optim.SGD(m.parameters(), lr=1e-3)
    sc = torch.amp.GradScaler("cuda")
    # Step 1: normal
    d.start_step(); opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.float16):
        y = m(torch.randn(32,256, device=DEV)); loss = (y**2).mean()
    sc.scale(loss).backward(); d.collect_signals(loss, m); sc.step(opt); sc.update(); d.end_step(opt, sc)
    # Step 2: cause overflow (exaggerate loss)
    d.start_step(); opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.float16):
        y = m(torch.randn(32,256, device=DEV)); loss = (y**2).mean() * 1e8
    sc.scale(loss).backward(); d.collect_signals(loss, m); sc.step(opt); sc.update(); d.end_step(opt, sc)
    # Next step’s mode should be fp32 due to scale drop
    assert all(st.mode == "fp32" for st in d._registry.values())
