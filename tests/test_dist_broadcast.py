import socket
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Optional

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.multiprocessing.spawn import ProcessRaisedException

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dpcs.runtime import dist_broadcast

pytestmark = pytest.mark.skipif(
    not torch.distributed.is_available(),
    reason="torch.distributed is not available",
)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _broadcast_worker(rank: int, world_size: int, port: int,
                      skip_rank: Optional[int], skip_step: Optional[int], steps: int) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        world_size=world_size,
        rank=rank,
        timeout=timedelta(seconds=3),
    )
    try:
        tensor = torch.zeros(1, dtype=torch.float32)
        for step in range(steps):
            if skip_rank is not None and rank == skip_rank and skip_step == step:
                time.sleep(0.5)
                continue
            if rank == 0:
                tensor.fill_(float(step + 1))
            else:
                tensor.zero_()
            dist_broadcast(tensor)
            assert pytest.approx(float(step + 1)) == float(tensor.item())
    finally:
        dist.destroy_process_group()


def test_dist_broadcast_all_ranks_participate():
    port = _find_free_port()
    mp.spawn(
        _broadcast_worker,
        args=(2, port, None, None, 3),
        nprocs=2,
        join=True,
    )


def test_dist_broadcast_missing_rank_fails_fast():
    port = _find_free_port()
    with pytest.raises(ProcessRaisedException) as excinfo:
        mp.spawn(
            _broadcast_worker,
            args=(2, port, 1, 1, 2),
            nprocs=2,
            join=True,
        )
    message = str(excinfo.value)
    assert "dist_broadcast" in message
    assert "participation" in message or "barrier" in message
