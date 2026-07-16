"""Compatibility shim for ``seiso.slime``.

Prefer ``from seiso.slime import SlimeConfig`` (or ``SingleGpuSlimeConfig``).
This package re-exports the same modules so existing imports keep working.
"""

from __future__ import annotations

from seiso.slime.config import SingleGpuSlimeConfig, SlimeConfig

__all__ = ["SlimeConfig", "SingleGpuSlimeConfig"]
