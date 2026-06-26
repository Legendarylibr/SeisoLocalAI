"""Model-aware context window limits for chat inference."""

from __future__ import annotations

import json
import re
from pathlib import Path

# Presets shown in the chat UI (filtered to effective max).
CONTEXT_WINDOW_PRESETS: tuple[int, ...] = (
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
    131072,
)

# Hard ceiling — never exceed even when model metadata claims more.
ABSOLUTE_MAX_CTX = 131072

_DEFAULT_UNKNOWN_CTX = 8192

_CTX_NAME_RE = re.compile(r"(?:^|[-_.])(?:(\d+)k|(\d{5,6}))\b", re.I)


def hf_config_context_length(model_path: str) -> int | None:
    """Read max context from a local Hugging Face model directory."""
    path = Path(model_path)
    root = path.parent if path.is_file() else path
    config_path = root / "config.json"
    if not config_path.is_file():
        return None
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(data, dict):
        return None

    for key in ("max_position_embeddings", "model_max_length", "n_positions", "seq_length"):
        value = data.get(key)
        if isinstance(value, int) and value >= 512:
            return value

    rope = data.get("rope_scaling")
    if isinstance(rope, dict):
        original = rope.get("original_max_position_embeddings")
        factor = rope.get("factor", 1)
        if isinstance(original, int) and isinstance(factor, (int, float)) and factor > 0:
            return int(original * factor)
    return None


def _context_from_name(text: str) -> int | None:
    for match in _CTX_NAME_RE.finditer(text.lower()):
        if match.group(1):
            return int(match.group(1)) * 1024
        if match.group(2):
            return int(match.group(2))
    return None


def resolve_model_context_ceiling(
    model_path: str | None,
    *,
    model_format: str | None = None,
    model_name: str | None = None,
) -> int:
    """Native/training context length before VRAM clamping."""
    if not model_path:
        return _DEFAULT_UNKNOWN_CTX

    path = Path(model_path)
    fmt = (model_format or "").lower()
    native: int | None = None

    if (
        fmt == "gguf"
        or path.suffix.lower() == ".gguf"
        or (path.is_file() and path.name.lower().endswith(".gguf"))
    ):
        from seiso.inference.backends import gguf_context_length

        native = gguf_context_length(model_path)
    if native is None and (fmt in {"safetensors", "bin", ""} or path.is_dir()):
        native = hf_config_context_length(model_path)
    if native is None:
        native = _context_from_name(path.name) or _context_from_name(model_name or "")

    if native is None:
        return _DEFAULT_UNKNOWN_CTX
    return max(2048, min(int(native), ABSOLUTE_MAX_CTX))


def effective_context_ceiling(
    model_path: str | None = None,
    *,
    model_format: str | None = None,
    model_name: str | None = None,
) -> int:
    """Maximum selectable context after model capability and free VRAM."""
    from seiso.memory.protection import (
        _INFERENCE_OVERHEAD_MB,
        _MIN_LLAMA_CTX,
        headroom_mb,
    )

    step = 512
    headroom = headroom_mb()
    vram_cap = max(_MIN_LLAMA_CTX, int((headroom - _INFERENCE_OVERHEAD_MB) * 5))
    vram_cap = min(vram_cap, ABSOLUTE_MAX_CTX)
    vram_cap = (vram_cap // step) * step or _MIN_LLAMA_CTX

    model_cap = resolve_model_context_ceiling(
        model_path,
        model_format=model_format,
        model_name=model_name,
    )
    return min(model_cap, vram_cap)


def context_window_presets(max_ctx: int) -> list[int]:
    """UI dropdown values up to the effective ceiling."""
    max_ctx = max(2048, int(max_ctx))
    presets = [value for value in CONTEXT_WINDOW_PRESETS if value <= max_ctx]
    if max_ctx not in presets and max_ctx >= 2048:
        presets.append(max_ctx)
        presets.sort()
    return presets or [2048]
