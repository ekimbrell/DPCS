"""
Dynamic Precision & Checkpointing Scheduler (DPCS) public API
Usage:
    from dpcs import DPCS, DPCSConfig
"""
from __future__ import annotations

from .scheduler import DPCS          # orchestrator
from .config import DPCSConfig       # frozen config

__all__ = ["DPCS", "DPCSConfig"]
