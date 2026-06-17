"""Bootstrap vendored sd_compress onto sys.path."""

from __future__ import annotations

import sys
from pathlib import Path

_VENDOR_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "sd-distill-prune-quant"


def vendor_root() -> Path:
    return _VENDOR_ROOT


def ensure_sd_compress_importable() -> Path:
    root = str(_VENDOR_ROOT.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    return _VENDOR_ROOT


def require_sd_compress() -> None:
    ensure_sd_compress_importable()
    try:
        import sd_compress  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Stable Diffusion compression vendor missing. "
            "Expected third_party/sd-distill-prune-quant/sd_compress"
        ) from exc
