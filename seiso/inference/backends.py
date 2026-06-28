"""Inference backend selection for local models."""

from __future__ import annotations

import re
import struct
from pathlib import Path

from seiso.compat import StrEnum
from seiso.models.loader import Backend, detect_backend

BackendName = str


class InferenceBackend(StrEnum):
    LLAMACPP = "llamacpp"
    MLX = "mlx"
    TORCH = "torch"
    ROUTER = "router"
    AUTO = "auto"


BACKEND_LLAMACPP = InferenceBackend.LLAMACPP
BACKEND_MLX = InferenceBackend.MLX
BACKEND_TORCH = InferenceBackend.TORCH
BACKEND_ROUTER = InferenceBackend.ROUTER
BACKEND_AUTO = InferenceBackend.AUTO

BACKEND_LABELS: dict[str, str] = {
    "llamacpp": "llama.cpp",
    "mlx": "MLX",
    "torch": "PyTorch",
    "router": "Smart Router",
    "auto": "Auto",
}
_GGUF_SHARD_RE = re.compile(r"^(?P<prefix>.+)-(?P<index>\d{5})-of-(?P<total>\d{5})\.gguf$", re.I)
_GGUF_VALUE_SIZE = {
    0: 1,  # uint8
    1: 1,  # int8
    2: 2,  # uint16
    3: 2,  # int16
    4: 4,  # uint32
    5: 4,  # int32
    6: 4,  # float32
    7: 1,  # bool
    10: 8,  # uint64
    11: 8,  # int64
    12: 8,  # float64
}
# dflash-draft are specialized tiny draft models. We allow llama.cpp backend for them
# (especially when used as speculative drafts). They are filtered from main catalogs
# via other hints.
_UNSUPPORTED_GGUF_ARCHITECTURES = frozenset()  # was {"dflash-draft"} - now supported as drafts


def is_dflash_draft(model_path: str) -> bool:
    """Detect if a GGUF path is a dflash/draft model (specialized small draft for speculative decoding)."""
    arch = gguf_architecture(model_path)
    if arch and "dflash" in arch.lower():
        return True
    name = Path(model_path).name.lower()
    return "dflash" in name or "-draft" in name or "draft" in name and "gguf" in name


def _looks_like_gguf_file(path: Path) -> bool:
    if path.suffix.lower() == ".gguf":
        return True
    if not path.is_file():
        return False
    try:
        with path.open("rb") as handle:
            return handle.read(4) == b"GGUF"
    except OSError:
        return False


def _is_gguf_path(model_path: str) -> bool:
    path = Path(model_path)
    if _looks_like_gguf_file(path):
        return True
    return path.is_dir() and any(path.glob("*.gguf"))


def resolve_gguf_file(model_path: str) -> Path:
    """Pick a single GGUF file from a path or directory."""
    path = Path(model_path).expanduser()
    if path.is_file() and _looks_like_gguf_file(path):
        return path.absolute()
    if path.is_dir():
        candidates = sorted(path.glob("*.gguf"))
        first_shards = [
            p
            for p in candidates
            if (match := _GGUF_SHARD_RE.match(p.name)) and match.group("index") == "00001"
        ]
        if first_shards:
            return first_shards[0].absolute()
        candidates = sorted(candidates, key=lambda p: p.stat().st_size, reverse=True)
        if candidates:
            return candidates[0].absolute()
    raise ValueError(f"No GGUF file found at {model_path}")


def _read_gguf_string(handle) -> str:
    raw_len = handle.read(8)
    if len(raw_len) != 8:
        raise ValueError("truncated GGUF string length")
    (length,) = struct.unpack("<Q", raw_len)
    raw = handle.read(length)
    if len(raw) != length:
        raise ValueError("truncated GGUF string")
    return raw.decode("utf-8", errors="replace")


def _skip_gguf_value(handle, value_type: int) -> None:
    if value_type == 8:  # string
        length = struct.unpack("<Q", handle.read(8))[0]
        handle.seek(length, 1)
        return
    if value_type == 9:  # array
        raw = handle.read(12)
        if len(raw) != 12:
            raise ValueError("truncated GGUF array header")
        item_type, count = struct.unpack("<IQ", raw)
        if item_type == 8:
            for _ in range(count):
                _skip_gguf_value(handle, item_type)
            return
        item_size = _GGUF_VALUE_SIZE.get(item_type)
        if item_size is None:
            raise ValueError(f"unsupported GGUF array type: {item_type}")
        handle.seek(item_size * count, 1)
        return
    size = _GGUF_VALUE_SIZE.get(value_type)
    if size is None:
        raise ValueError(f"unsupported GGUF value type: {value_type}")
    handle.seek(size, 1)


_gguf_arch_cache: dict[tuple[str, float, int], str | None] = {}
_gguf_block_count_cache: dict[tuple[str, float, int], int | None] = {}
_gguf_context_length_cache: dict[tuple[str, float, int], int | None] = {}


def clear_gguf_caches() -> None:
    """Reset GGUF architecture cache (for tests)."""
    _gguf_arch_cache.clear()
    _gguf_block_count_cache.clear()
    _gguf_context_length_cache.clear()


def _gguf_cache_key(path: Path) -> tuple[str, float, int] | None:
    try:
        stat = path.stat()
        return (str(path), stat.st_mtime, stat.st_size)
    except OSError:
        return None


def gguf_architecture(model_path: str) -> str | None:
    """Read ``general.architecture`` from a GGUF file when available."""
    try:
        path = resolve_gguf_file(model_path)
    except ValueError:
        return None

    cache_key = _gguf_cache_key(path)
    if cache_key is not None:
        cached = _gguf_arch_cache.get(cache_key)
        if cached is not None or cache_key in _gguf_arch_cache:
            return cached

    architecture: str | None
    try:
        architecture = _read_gguf_architecture(path)
    except (OSError, ValueError, struct.error):
        architecture = None

    if cache_key is not None:
        _gguf_arch_cache[cache_key] = architecture
    return architecture


def _gguf_has_metadata_key(path: Path, key_suffix: str) -> bool:
    with path.open("rb") as handle:
        if handle.read(4) != b"GGUF":
            return False
        header = handle.read(20)
        if len(header) != 20:
            return False
        _version, _tensor_count, kv_count = struct.unpack("<IQQ", header)
        for _ in range(kv_count):
            key = _read_gguf_string(handle)
            raw_type = handle.read(4)
            if len(raw_type) != 4:
                return False
            (value_type,) = struct.unpack("<I", raw_type)
            if key.endswith(key_suffix):
                return True
            _skip_gguf_value(handle, value_type)
    return False


def _read_gguf_metadata_u32(path: Path, key_suffix: str) -> int | None:
    with path.open("rb") as handle:
        if handle.read(4) != b"GGUF":
            return None
        header = handle.read(20)
        if len(header) != 20:
            return None
        _version, _tensor_count, kv_count = struct.unpack("<IQQ", header)
        for _ in range(kv_count):
            key = _read_gguf_string(handle)
            raw_type = handle.read(4)
            if len(raw_type) != 4:
                return None
            (value_type,) = struct.unpack("<I", raw_type)
            if key.endswith(key_suffix) and value_type == 4:
                raw = handle.read(4)
                if len(raw) != 4:
                    return None
                (value,) = struct.unpack("<I", raw)
                return int(value)
            _skip_gguf_value(handle, value_type)
    return None


def gguf_context_length(model_path: str) -> int | None:
    """Read training context length from GGUF metadata (e.g. llama.context_length)."""
    try:
        path = resolve_gguf_file(model_path)
    except ValueError:
        return None

    cache_key = _gguf_cache_key(path)
    if cache_key is not None and cache_key in _gguf_context_length_cache:
        return _gguf_context_length_cache[cache_key]

    length: int | None
    try:
        length = _read_gguf_metadata_u32(path, ".context_length")
    except (OSError, ValueError, struct.error):
        length = None

    if cache_key is not None:
        _gguf_context_length_cache[cache_key] = length
    return length


def gguf_uses_sliding_window_attention(model_path: str) -> bool:
    """True when GGUF metadata indicates sliding-window attention layers."""
    try:
        path = resolve_gguf_file(model_path)
    except ValueError:
        return False

    try:
        sliding_window = _read_gguf_metadata_u32(path, ".attention.sliding_window")
        if sliding_window is not None and sliding_window > 0:
            return True
        return _gguf_has_metadata_key(path, ".attention.sliding_window_pattern")
    except (OSError, ValueError, struct.error):
        return False


def gguf_block_count(model_path: str) -> int | None:
    """Read transformer block count from GGUF metadata when available."""
    try:
        path = resolve_gguf_file(model_path)
    except ValueError:
        return None

    cache_key = _gguf_cache_key(path)
    if cache_key is not None and cache_key in _gguf_block_count_cache:
        return _gguf_block_count_cache[cache_key]

    count: int | None
    try:
        count = _read_gguf_metadata_u32(path, ".block_count")
    except (OSError, ValueError, struct.error):
        count = None

    if cache_key is not None:
        _gguf_block_count_cache[cache_key] = count
    return count


def _read_gguf_architecture(path: Path) -> str | None:
    with path.open("rb") as handle:
        if handle.read(4) != b"GGUF":
            return None
        header = handle.read(20)
        if len(header) != 20:
            return None
        _version, _tensor_count, kv_count = struct.unpack("<IQQ", header)
        for _ in range(kv_count):
            key = _read_gguf_string(handle)
            raw_type = handle.read(4)
            if len(raw_type) != 4:
                return None
            (value_type,) = struct.unpack("<I", raw_type)
            if key == "general.architecture" and value_type == 8:
                return _read_gguf_string(handle)
            _skip_gguf_value(handle, value_type)
    return None


def gguf_is_supported_by_llamacpp(model_path: str) -> bool:
    architecture = gguf_architecture(model_path)
    return architecture not in _UNSUPPORTED_GGUF_ARCHITECTURES


def recommend_backend(*, model_path: str, model_format: str | None = None) -> BackendName:
    """Pick the default local inference engine from model path/format."""
    fmt = (model_format or "").lower()
    path = Path(model_path)
    if fmt == "gguf" or _is_gguf_path(model_path):
        return BACKEND_LLAMACPP
    if fmt in {"safetensors", "bin"} or path.is_dir():
        backend = detect_backend()
        if backend == Backend.MLX:
            return BACKEND_MLX
        return BACKEND_TORCH
    if path.suffix.lower() == ".gguf":
        return BACKEND_LLAMACPP
    return BACKEND_TORCH


def available_backends(*, model_path: str, model_format: str | None = None) -> list[BackendName]:
    """Backends that can serve this inventory model."""
    if (model_format or "").lower() == "gguf" and not gguf_is_supported_by_llamacpp(model_path):
        return []
    return [recommend_backend(model_path=model_path, model_format=model_format)]


def resolve_local_backend(
    *,
    model_path: str,
    model_format: str | None,
    requested: str | None,
) -> BackendName:
    """Resolve auto/requested backend for a filesystem model."""
    choice = (requested or BACKEND_AUTO).lower()
    if choice == BACKEND_AUTO:
        return recommend_backend(model_path=model_path, model_format=model_format)
    if choice in {BACKEND_LLAMACPP, BACKEND_MLX, BACKEND_TORCH}:
        return choice
    raise ValueError(f"Unsupported inference backend: {requested}")


def prepare_model_path(model_path: str, backend: BackendName) -> str:
    """Normalize model path (e.g. pick a GGUF file inside a directory)."""
    if backend == BACKEND_LLAMACPP:
        return str(resolve_gguf_file(model_path))
    return model_path
