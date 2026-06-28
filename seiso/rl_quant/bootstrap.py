"""Bootstrap bundled seiso.adaptive_quant onto sys.path."""

from __future__ import annotations

from pathlib import Path

from seiso.bundled.packages import ADAPTIVE_QUANT

_BUNDLE_ROOT = ADAPTIVE_QUANT.root


def bundle_root() -> Path:
    return _BUNDLE_ROOT


ensure_adaptive_quant_importable = ADAPTIVE_QUANT.ensure_importable
require_adaptive_quant = ADAPTIVE_QUANT.require
