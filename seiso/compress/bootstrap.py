"""Bootstrap vendored codellama_compress onto sys.path."""

from __future__ import annotations

from pathlib import Path

from seiso.vendor.bootstrap import make_vendor_bootstrap

_bs = make_vendor_bootstrap(
    "codellama-compress",
    "codellama_compress",
    missing_hint="Expected third_party/codellama-compress/src/codellama_compress",
)
_VENDOR_ROOT = _bs.root


def vendor_root() -> Path:
    return _VENDOR_ROOT


ensure_codellama_compress_importable = _bs.ensure_importable
require_codellama_compress = _bs.require
