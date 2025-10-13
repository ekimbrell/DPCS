import importlib
import importlib.util
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest


def _import_scheduler_module():
    try:
        return importlib.import_module("dpcs.scheduler")
    except Exception:
        here = Path(__file__).resolve().parents[1]
        package_init = here / "src" / "dpcs" / "__init__.py"
        if package_init.exists():
            pkg_spec = importlib.util.spec_from_file_location("dpcs", str(package_init))
            pkg = importlib.util.module_from_spec(pkg_spec)
            assert pkg_spec and pkg_spec.loader
            sys.modules.setdefault("dpcs", pkg)
            pkg_spec.loader.exec_module(pkg)  # type: ignore[attr-defined]
        candidate = here / "src" / "dpcs" / "scheduler.py"
        if not candidate.exists():
            raise
        spec = importlib.util.spec_from_file_location("dpcs.scheduler", str(candidate))
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
        return module


SCHEDULER_MOD = _import_scheduler_module()
DPCS = SCHEDULER_MOD.DPCS


class TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.layers(x)


def _run_training(tmp_path: Path, *, compile_diagnostics: bool):
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile is required for this test")

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TinyMLP().to(device)
    scheduler = DPCS(
        compile_diagnostics=compile_diagnostics,
        compile_warmup_steps=2,
        no_flip_during_warmup=True,
        log_every=1,
    )
    model = scheduler.wrap(model).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer.zero_grad(set_to_none=True)

    batches = []
    for _ in range(4):
        x = torch.randn(8, 4, device=device)
        y = torch.randn(8, 2, device=device)
        batches.append((x, y))

    def forward_loss(mod: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        out = mod(x)
        return F.mse_loss(out, y)

    compiled_forward = torch.compile(forward_loss)

    log_path = tmp_path / ("diag.jsonl" if compile_diagnostics else "plain.jsonl")
    scheduler.set_log_jsonl(str(log_path))

    if compile_diagnostics:
        sample_x, sample_y = batches[0][0].clone(), batches[0][1].clone()
        scheduler.set_compile_step_fn(compiled_forward, model, sample_x, sample_y)

    scheduler._prec_pol.decide = lambda *args, **kwargs: "fp16"  # type: ignore[assignment]
    scheduler._ckpt_pol.plan = lambda *args, **kwargs: [0]  # type: ignore[assignment]

    modes = []
    ckpt_history = []

    for x, y in batches:
        scheduler.start_step()
        loss = compiled_forward(model, x, y)
        scheduler.collect_signals(loss, model)
        loss.backward()
        scheduler.end_step(optimizer)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        modes.append(scheduler._amp_mode)
        ckpt_history.append(set(scheduler._ckpt_selected))

    scheduler.set_log_jsonl("")

    data = []
    if log_path.exists():
        with open(log_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    data.append(json.loads(line))

    return modes, ckpt_history, data


def test_compile_diagnostics_warmup(tmp_path):
    modes_diag, ckpts_diag, log_diag = _run_training(tmp_path, compile_diagnostics=True)
    modes_plain, ckpts_plain, log_plain = _run_training(tmp_path, compile_diagnostics=False)

    assert modes_diag[0] == modes_diag[1]
    assert modes_diag[2] == "fp16"
    assert ckpts_diag[0] == set()
    assert ckpts_diag[1] == set()
    assert ckpts_diag[2] == {0}

    assert modes_plain[0] == "fp16"
    assert ckpts_plain[0] == {0}

    assert any("graph_breaks_total" in rec for rec in log_diag)
    assert all("graph_breaks_total" not in rec for rec in log_plain)
