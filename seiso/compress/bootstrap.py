"""Bootstrap bundled seiso.codellama_compress onto sys.path."""

from __future__ import annotations

from pathlib import Path

from seiso.vendor.packages import CODELLAMA

_VENDOR_ROOT = CODELLAMA.root


def vendor_root() -> Path:
    return _VENDOR_ROOT


ensure_codellama_compress_importable = CODELLAMA.ensure_importable
require_codellama_compress = CODELLAMA.require
