"""Backward-compatibility shim for importing ``dpcs.dpcs``.

The legacy layout exposed ``DPCS`` and ``DPCSConfig`` from this module. The
new package keeps the orchestrator in ``scheduler.py`` but tests and external
scripts may still import :mod:`dpcs.dpcs`, so re-export the public API here.
"""
from __future__ import annotations

from .scheduler import DPCS
from .config import DPCSConfig

__all__ = ["DPCS", "DPCSConfig"]
