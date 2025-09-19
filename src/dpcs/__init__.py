"""
Dynamic Precision & Checkpointing Scheduler (DPCS) public API
Usage:
    from dpcs import DPCS, DPCSConfig
"""
from __future__ import annotations

from .scheduler import DPCS          # orchestrator
from .config import DPCSConfig, CheckpointCfg  # frozen configs

__all__ = ["DPCS", "DPCSConfig", "CheckpointCfg"]
