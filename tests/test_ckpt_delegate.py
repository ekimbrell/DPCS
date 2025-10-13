import types

import pytest
import torch
import torch.nn as nn

from dpcs import DPCS


class TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


def _is_cuda():
    return torch.cuda.is_available()


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if _is_cuda() else []))
def test_delegate_mutual_exclusion(device):
    model = TinyNet().to(device).train()
    sched = DPCS(
        device_type=device,
        delegate_selective_ckpt=True,
        activation_memory_budget_frac=0.4,
        enable_precision=False,
    )
    model = sched.wrap(model)

    with pytest.raises(RuntimeError):
        sched.enable_checkpointing(True)

    sched.enable_checkpointing(False)

    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    x = torch.randn(4, 8, device=device)
    target = torch.randn(4, 8, device=device)

    sched.start_step()
    opt.zero_grad(set_to_none=True)
    with sched.forward_context():
        out = model(x)
    loss = torch.nn.functional.mse_loss(out, target)
    loss.backward()
    sched.collect_signals(loss, model)
    opt.step()
    sched.end_step(opt)

    assert not sched._ckpt_selected
    assert all(not leaf.use_ckpt for leaf in sched._leaves)


def test_forward_context_budget_toggle(monkeypatch):
    device = "cpu"
    sched = DPCS(
        device_type=device,
        delegate_selective_ckpt=True,
        activation_memory_budget_frac=0.3,
        enable_precision=False,
    )

    holder = types.SimpleNamespace(activation_memory_budget=0.0)
    monkeypatch.setattr(torch, "_functorch", types.SimpleNamespace(config=holder), raising=False)

    with sched.forward_context():
        assert holder.activation_memory_budget == pytest.approx(0.3, rel=1e-6)

    assert holder.activation_memory_budget == 0.0

    sched_non_delegate = DPCS(device_type=device, enable_precision=False)
    with sched_non_delegate.forward_context():
        assert holder.activation_memory_budget == 0.0
