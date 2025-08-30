import math
import importlib
import importlib.util
from pathlib import Path

import pytest
import os as _os
_os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
import torch.nn as nn

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for DPCS tests")


# --- Robust import of the module under test ---------------------------------
# Works if package is installed (pip install -e .) OR running against src/ layout.

def _import_dpcs_module():
    # 0) Try the installed package path first
    try:
        return importlib.import_module("dpcs.dpcs")
    except Exception:
        pass

    # 1) Try to import from src/ layout
    here = Path(__file__).resolve().parents[1]  # repo root
    candidate = here / "src" / "dpcs" / "dpcs.py"
    if candidate.exists():
        spec = importlib.util.spec_from_file_location("dpcs.dpcs", str(candidate))
        mod = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]
        return mod

    # 2) Fallback to a top-level file (flat layout)
    flat = here / "dpcs.py"
    if flat.exists():
        spec = importlib.util.spec_from_file_location("dpcs", str(flat))
        mod = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]
        return mod

    raise RuntimeError("Could not locate DPCS module. Install the package or keep it under src/dpcs/dpcs.py")


_dpcs = _import_dpcs_module()
DPCS = _dpcs.DPCS
DPCSConfig = _dpcs.DPCSConfig


# --- Helpers ----------------------------------------------------------------

def tiny_mlp(n_in=1024, n_hidden=1024, n_out=1024, depth=2):
    layers = []
    for _ in range(depth):
        layers.append(nn.Sequential(nn.Linear(n_in, n_hidden), nn.GELU(), nn.Linear(n_hidden, n_out)))
    return nn.Sequential(*layers).cuda().train()


def modes_summary(scheduler: "DPCS"):
    counts = {}
    for st in scheduler._registry.values():  # test may use private internals
        counts[st.mode] = counts.get(st.mode, 0) + 1
    return counts


def run_step(scheduler: "DPCS", model: nn.Module, optimizer, use_scaler: bool):
    scheduler.start_step()
    x = torch.randn(4, 1024, device="cuda", dtype=torch.float32)
    y = model(x)
    loss = (y.float() ** 2).mean()

    if use_scaler:
        scaler = torch.amp.GradScaler("cuda")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.end_step(optimizer, scaler)
    else:
        loss.backward()
        optimizer.step()
        scheduler.end_step(optimizer, None)

    optimizer.zero_grad(set_to_none=True)
    return float(loss.detach().item())


# --- Tests ------------------------------------------------------------------

def test_milestone1_amp_dtype_and_overflow_cooldown():
    cfg = dict(
        enable_precision=True,
        cooldown_steps=2,
        wrap_types=(nn.Linear, nn.Sequential),
        ckpt_wrap_types=(nn.Sequential,),
    )
    model = tiny_mlp(depth=2)
    sched = DPCS(**cfg)
    model = sched.wrap(model)
    optim = torch.optim.SGD(model.parameters(), lr=1e-3)

    use_scaler = sched.amp_dtype() == torch.float16

    # Warmup a couple of steps under AMP
    for _ in range(2):
        run_step(sched, model, optim, use_scaler)

    # Simulate a GradScaler scale drop (overflow signal) by calling end_step with a fake scaler
    class FakeScaler:
        def __init__(self, scale):
            self._scale = scale
        def get_scale(self):
            return float(self._scale)

    # Establish a baseline scale, then a lower one to trigger cooldown
    sched.end_step(optim, FakeScaler(1024.0))
    sched.end_step(optim, FakeScaler(512.0))  # drop => cooldown should engage

    mix = modes_summary(sched)
    assert mix.get("fp32", 0) >= 1, f"Expected some modules in fp32 after cooldown, got {mix}"

    # Cooldown counters should be set on wrapped modules
    cools = [st.cool for st in sched._registry.values()]
    assert all(c >= sched.cfg.cooldown_steps for c in cools), f"Cooldown not set correctly: {cools}"


def test_milestone2_topk_checkpoint_selection_and_blacklist():
    # Configure so our tiny model has eligible candidates and blacklisting trips deterministically
    cfg = dict(
        enable_precision=True,
        cooldown_steps=1,
        wrap_types=(nn.Sequential, nn.Linear),
        ckpt_wrap_types=(nn.Sequential,),
        min_activation_bytes_to_ckpt=1 << 10,
        ckpt_topk_frac=0.5,
        ckpt_min_candidates=1,
        ckpt_max_blocks=None,
        ckpt_harmful_delta_bytes=-1,  # any positive delta marks harmful
        ckpt_harmful_patience=1,
    )
    model = tiny_mlp(depth=6)
    sched = DPCS(**cfg)
    sched.enable_checkpointing(True)
    model = sched.wrap(model)
    optim = torch.optim.SGD(model.parameters(), lr=1e-3)

    # Warmup to populate activation byte estimates
    run_step(sched, model, optim, use_scaler=(sched.amp_dtype() == torch.float16))

    # Force low headroom so selection fraction rises (Milestone 3 influence on Milestone 2 selection)
    sched.vram_headroom = lambda: 0.05
    sched.start_step()

    # Count eligible Sequential modules present in registry
    seq_modules = [m for m in model.modules() if isinstance(m, nn.Sequential) and m in sched._registry]
    num_cands = len([m for m in seq_modules if not sched._registry[m].ckpt_blacklisted])

    # Expected K after low-headroom boost (1.5x, clipped to 1.0)
    frac = min(1.0, sched.cfg.ckpt_topk_frac * 1.5)
    expected_k = max(int(math.ceil(frac * num_cands)), int(sched.cfg.ckpt_min_candidates))
    if sched.cfg.ckpt_max_blocks is not None:
        expected_k = min(expected_k, int(sched.cfg.ckpt_max_blocks))

    assert len(sched._ckpt_selected) == expected_k, (
        len(sched._ckpt_selected), expected_k, num_cands
    )

    # Run a backward/step so harmful detection runs and blacklists the selected modules
    x = torch.randn(2, 1024, device="cuda")
    (model(x).sum()).backward()
    optim.step(); optim.zero_grad(set_to_none=True)
    sched.end_step(optim, None)

    # Verify that at least one of the previously selected modules is now blacklisted
    selected_list = list(sched._ckpt_selected)
    blacklisted = [m for m in selected_list if sched._registry[m].ckpt_blacklisted]
    assert len(blacklisted) >= 1, "Expected some modules to be blacklisted after harmful checkpoint detection"


def test_milestone3_headroom_bias_changes_selection_size():
    cfg = dict(
        enable_precision=True,
        cooldown_steps=1,
        wrap_types=(nn.Sequential, nn.Linear),
        ckpt_wrap_types=(nn.Sequential,),
        min_activation_bytes_to_ckpt=1 << 10,
        ckpt_topk_frac=0.25,
        ckpt_min_candidates=1,
        ckpt_max_blocks=None,
        ckpt_harmful_delta_bytes=1 << 60,  # practically never harmful in this test
        ckpt_harmful_patience=100,
    )
    model = tiny_mlp(depth=8)
    sched = DPCS(**cfg)
    sched.enable_checkpointing(True)
    model = sched.wrap(model)
    optim = torch.optim.SGD(model.parameters(), lr=1e-3)

    # Warmup to get activation sizes
    run_step(sched, model, optim, use_scaler=(sched.amp_dtype() == torch.float16))

    # Comfortable headroom => baseline selection size
    sched.vram_headroom = lambda: 0.5
    sched.start_step()
    baseline = len(sched._ckpt_selected)

    # Low headroom => selection should increase or stay the same
    sched._ckpt_selected.clear()
    sched.vram_headroom = lambda: 0.05
    sched.start_step()
    pressure = len(sched._ckpt_selected)

    assert pressure >= baseline, f"Expected >= checkpointed modules under pressure ({pressure} vs {baseline})"
