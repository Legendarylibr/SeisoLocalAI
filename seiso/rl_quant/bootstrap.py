"""Bootstrap vendored adaptive_quant onto sys.path."""

from __future__ import annotations

from pathlib import Path

from seiso.vendor.packages import ADAPTIVE_QUANT

_VENDOR_ROOT = ADAPTIVE_QUANT.root


def vendor_root() -> Path:
    return _VENDOR_ROOT


ensure_adaptive_quant_importable = ADAPTIVE_QUANT.ensure_importable
require_adaptive_quant = ADAPTIVE_QUANT.require
