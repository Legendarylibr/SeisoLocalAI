"""Bootstrap vendored adaptive_quant onto sys.path."""

from __future__ import annotations

from pathlib import Path

from seiso.vendor.bootstrap import make_vendor_bootstrap

_bs = make_vendor_bootstrap(
    "adaptive-rl-quant",
    "adaptive_quant",
    missing_hint="Expected third_party/adaptive-rl-quant/src/adaptive_quant",
)
_VENDOR_ROOT = _bs.root


def vendor_root() -> Path:
    return _VENDOR_ROOT


ensure_adaptive_quant_importable = _bs.ensure_importable
require_adaptive_quant = _bs.require
