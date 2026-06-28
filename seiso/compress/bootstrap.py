"""Bootstrap bundled seiso.codellama_compress onto sys.path."""

from __future__ import annotations

from pathlib import Path

from seiso.bundled.packages import CODELLAMA

_BUNDLE_ROOT = CODELLAMA.root


def bundle_root() -> Path:
    return _BUNDLE_ROOT


ensure_codellama_compress_importable = CODELLAMA.ensure_importable
require_codellama_compress = CODELLAMA.require
