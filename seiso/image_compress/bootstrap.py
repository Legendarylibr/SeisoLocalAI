"""Bootstrap vendored sd_compress onto sys.path."""

from __future__ import annotations

from pathlib import Path

from seiso.vendor.bootstrap import ensure_vendor_importable, require_vendor_package

_VENDOR_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "sd-distill-prune-quant"


def vendor_root() -> Path:
    return _VENDOR_ROOT


def ensure_sd_compress_importable() -> Path:
    return ensure_vendor_importable(_VENDOR_ROOT, src_subdir=None)


def require_sd_compress() -> None:
    require_vendor_package(
        _VENDOR_ROOT,
        "sd_compress",
        src_subdir=None,
        missing_hint="Expected third_party/sd-distill-prune-quant/sd_compress",
    )
