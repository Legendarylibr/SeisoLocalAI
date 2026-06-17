"""Bootstrap vendored codellama_compress onto sys.path."""

from __future__ import annotations

from pathlib import Path

from seiso.vendor.bootstrap import ensure_vendor_importable, require_vendor_package

_VENDOR_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "codellama-compress"


def vendor_root() -> Path:
    return _VENDOR_ROOT


def ensure_codellama_compress_importable() -> Path:
    return ensure_vendor_importable(_VENDOR_ROOT)


def require_codellama_compress() -> None:
    require_vendor_package(
        _VENDOR_ROOT,
        "codellama_compress",
        src_subdir="src",
        missing_hint="Expected third_party/codellama-compress/src/codellama_compress",
    )
