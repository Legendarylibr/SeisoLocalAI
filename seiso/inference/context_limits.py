"""Model-aware context window limits for chat inference."""

from __future__ import annotations

import re
from collections import OrderedDict
from pathlib import Path

from seiso.io.jsonl import read_json_file

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

_HF_CONFIG_CACHE_MAX = 64
_hf_config_cache: OrderedDict[tuple[str, float, int], int | None] = OrderedDict()


def _hf_config_cache_key(config_path: Path) -> tuple[str, float, int] | None:
    try:
        stat = config_path.stat()
        return (str(config_path), stat.st_mtime, stat.st_size)
    except OSError:
        return None


def hf_config_context_length(model_path: str) -> int | None:
    """Read max context from a local Hugging Face model directory."""
    path = Path(model_path)
    root = path.parent if path.is_file() else path
    config_path = root / "config.json"
    cache_key = _hf_config_cache_key(config_path)
    if cache_key is not None and cache_key in _hf_config_cache:
        _hf_config_cache.move_to_end(cache_key)
        return _hf_config_cache[cache_key]

    data = read_json_file(config_path, default=None)
    result = (
        None if not isinstance(data, dict) else _context_length_from_hf_config(data)
    )

    if cache_key is not None:
        _hf_config_cache[cache_key] = result
        _hf_config_cache.move_to_end(cache_key)
        while len(_hf_config_cache) > _HF_CONFIG_CACHE_MAX:
            _hf_config_cache.popitem(last=False)
    return result


def _context_length_from_hf_config(data: dict) -> int | None:
    for key in (
        "max_position_embeddings",
        "model_max_length",
        "n_positions",
        "seq_length",
    ):
        value = data.get(key)
        if isinstance(value, int) and value >= 512:
            return value

    rope = data.get("rope_scaling")
    if isinstance(rope, dict):
        original = rope.get("original_max_position_embeddings")
        factor = rope.get("factor", 1)
        if (
            isinstance(original, int)
            and isinstance(factor, (int, float))
            and factor > 0
        ):
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
    """Maximum selectable context from model capability only."""
    return resolve_model_context_ceiling(
        model_path,
        model_format=model_format,
        model_name=model_name,
    )


def context_window_presets(max_ctx: int) -> list[int]:
    """UI dropdown values up to the effective ceiling."""
    max_ctx = max(2048, int(max_ctx))
    presets = [value for value in CONTEXT_WINDOW_PRESETS if value <= max_ctx]
    if max_ctx not in presets and max_ctx >= 2048:
        presets.append(max_ctx)
        presets.sort()
    return presets or [2048]
