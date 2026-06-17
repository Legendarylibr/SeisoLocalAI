"""Bootstrap vendored adaptive_quant onto sys.path."""

from __future__ import annotations

from pathlib import Path

from seiso.vendor.bootstrap import ensure_vendor_importable, require_vendor_package

_VENDOR_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "adaptive-rl-quant"


def vendor_root() -> Path:
    return _VENDOR_ROOT


def ensure_adaptive_quant_importable() -> Path:
    return ensure_vendor_importable(_VENDOR_ROOT)


def require_adaptive_quant() -> None:
    require_vendor_package(
        _VENDOR_ROOT,
        "adaptive_quant",
        src_subdir="src",
        missing_hint="Expected third_party/adaptive-rl-quant/src/adaptive_quant",
    )
