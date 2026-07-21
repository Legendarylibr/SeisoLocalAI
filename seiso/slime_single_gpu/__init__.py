"""Deprecated compatibility shim for ``seiso.slime``.

Prefer ``from seiso.slime import SlimeConfig`` (or ``SingleGpuSlimeConfig``).
This package only re-exports ``seiso.slime`` modules so legacy imports keep
working. Do not add new code here; migrate callers to ``seiso.slime``.
"""

from __future__ import annotations

from seiso.slime.config import SingleGpuSlimeConfig, SlimeConfig

__all__ = ["SlimeConfig", "SingleGpuSlimeConfig"]
