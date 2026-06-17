"""Bootstrap vendored codellama_compress onto sys.path."""

from __future__ import annotations

import sys
from pathlib import Path

_VENDOR_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "codellama-compress"
_VENDOR_SRC = _VENDOR_ROOT / "src"


def vendor_root() -> Path:
    return _VENDOR_ROOT


def ensure_codellama_compress_importable() -> Path:
    root = str(_VENDOR_SRC.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    return _VENDOR_SRC


def require_codellama_compress() -> None:
    ensure_codellama_compress_importable()
    try:
        import codellama_compress  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Code Llama compression vendor missing. "
            "Expected third_party/codellama-compress/src/codellama_compress"
        ) from exc
