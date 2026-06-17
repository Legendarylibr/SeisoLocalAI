"""Bootstrap vendored adaptive_quant onto sys.path."""

from __future__ import annotations

import sys
from pathlib import Path

_VENDOR_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "adaptive-rl-quant"
_VENDOR_SRC = _VENDOR_ROOT / "src"


def vendor_root() -> Path:
    return _VENDOR_ROOT


def ensure_adaptive_quant_importable() -> Path:
    root = str(_VENDOR_SRC.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    return _VENDOR_SRC


def require_adaptive_quant() -> None:
    ensure_adaptive_quant_importable()
    try:
        import adaptive_quant  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Adaptive RL Quantization vendor missing. "
            "Expected third_party/adaptive-rl-quant/src/adaptive_quant"
        ) from exc
