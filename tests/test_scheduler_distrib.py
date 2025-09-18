import pytest
import torch

from dpcs import scheduler


class _FakeReduceOp:
    MIN = object()


class _FakeDist:
    ReduceOp = _FakeReduceOp
    ProcessGroup = object

    def __init__(self, reduced_value: float) -> None:
        self._value = float(reduced_value)

    def is_available(self) -> bool:
        return True

    def is_initialized(self) -> bool:
        return True

    def all_reduce(self, tensor: torch.Tensor, op=None) -> None:  # noqa: D401 - test stub
        tensor.fill_(self._value)


def test_scheduler_headroom_sync_min(monkeypatch):
    sched = scheduler.DPCS(device_type="cuda")

    base_headroom = 0.42
    shared_headroom = 0.13

    monkeypatch.setattr(scheduler, "headroom_frac", lambda: base_headroom)
    monkeypatch.setattr(torch, "distributed", _FakeDist(shared_headroom), raising=False)

    captured = {}

    def fake_decide(headroom, *args, **kwargs):
        captured["headroom"] = headroom
        return "fp32"

    monkeypatch.setattr(sched._prec_pol, "decide", fake_decide)

    class _DummyOptim:
        param_groups = []

    optim = _DummyOptim()

    sched.end_step(optim)

    assert "headroom" in captured
    assert captured["headroom"] == pytest.approx(shared_headroom)