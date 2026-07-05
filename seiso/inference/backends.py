"""Inference backend selection for local models."""

from __future__ import annotations

import mmap
import re
import struct
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from seiso.compat import StrEnum
from seiso.models.loader import Backend, detect_backend

BackendName = str


class InferenceBackend(StrEnum):
    LLAMACPP = "llamacpp"
    LLAMASWAP = "llamaswap"
    MLX = "mlx"
    TORCH = "torch"
    ROUTER = "router"
    AUTO = "auto"


BACKEND_LLAMACPP = InferenceBackend.LLAMACPP
BACKEND_LLAMASWAP = InferenceBackend.LLAMASWAP
BACKEND_MLX = InferenceBackend.MLX
BACKEND_TORCH = InferenceBackend.TORCH
BACKEND_ROUTER = InferenceBackend.ROUTER
BACKEND_AUTO = InferenceBackend.AUTO

BACKEND_LABELS: dict[str, str] = {
    "llamacpp": "llama.cpp",
    "llamaswap": "llama-swap",
    "mlx": "MLX",
    "torch": "PyTorch",
    "router": "Smart Router",
    "auto": "Auto",
}
_GGUF_SHARD_RE = re.compile(
    r"^(?P<prefix>.+)-(?P<index>\d{5})-of-(?P<total>\d{5})\.gguf$", re.I
)
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
_GGUF_TYPE_U32 = 4
_GGUF_TYPE_STRING = 8
_GGUF_TYPE_ARRAY = 9
_GGUF_MAGIC = b"GGUF"
_GGUF_HEADER_FMT = "<IQQ"
_GGUF_KEY_ARCH = b"general.architecture"
_GGUF_SUFFIX_CTX_LEN = b".context_length"
_GGUF_SUFFIX_BLOCK_CNT = b".block_count"
_GGUF_SUFFIX_SLIDING_WIN = b".attention.sliding_window"
_GGUF_SUFFIX_SLIDING_PAT = b".attention.sliding_window_pattern"
# dflash-draft are specialized tiny draft models. We allow llama.cpp backend for them
# (especially when used as speculative drafts). They are filtered from main catalogs
# via other hints.
_UNSUPPORTED_GGUF_ARCHITECTURES = (
    frozenset()
)  # was {"dflash-draft"} - now supported as drafts


@dataclass(slots=True)
class _GGUFMetadata:
    architecture: str | None = None
    context_length: int | None = None
    block_count: int | None = None
    has_sliding_window: bool = False


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
        first_shard: Path | None = None
        largest: tuple[int, str, Path] | None = None
        for candidate in path.glob("*.gguf"):
            match = _GGUF_SHARD_RE.match(candidate.name)
            if match and match.group("index") == "00001":
                if first_shard is None or candidate.name < first_shard.name:
                    first_shard = candidate
                continue
            try:
                size = candidate.stat().st_size
            except OSError:
                continue
            item = (size, candidate.name, candidate)
            if largest is None or item > largest:
                largest = item
        if first_shard is not None:
            return first_shard.absolute()
        if largest is not None:
            return largest[2].absolute()
    raise ValueError(f"No GGUF file found at {model_path}")


_GGUF_METADATA_CACHE_MAX_ENTRIES = 256
_gguf_metadata_cache: OrderedDict[tuple[str, float, int], _GGUFMetadata] = (
    OrderedDict()
)


def clear_gguf_caches() -> None:
    """Reset GGUF metadata cache (for tests)."""
    _gguf_metadata_cache.clear()


def _gguf_cache_key(path: Path) -> tuple[str, float, int] | None:
    try:
        stat = path.stat()
        return (str(path), stat.st_mtime, stat.st_size)
    except OSError:
        return None


def _skip_gguf_mmap_value(mm: mmap.mmap, offset: int, value_type: int) -> int:
    if value_type == _GGUF_TYPE_STRING:
        (length,) = struct.unpack_from("<Q", mm, offset)
        return offset + 8 + length
    if value_type == _GGUF_TYPE_ARRAY:
        item_type, count = struct.unpack_from("<IQ", mm, offset)
        offset += 12
        if item_type == _GGUF_TYPE_STRING:
            for _ in range(count):
                (length,) = struct.unpack_from("<Q", mm, offset)
                offset += 8 + length
            return offset
        item_size = _GGUF_VALUE_SIZE.get(item_type)
        if item_size is None:
            raise ValueError(f"unsupported GGUF array type: {item_type}")
        return offset + item_size * count
    size = _GGUF_VALUE_SIZE.get(value_type)
    if size is None:
        raise ValueError(f"unsupported GGUF value type: {value_type}")
    return offset + size


def _read_gguf_metadata(path: Path) -> _GGUFMetadata:
    meta = _GGUFMetadata()
    try:
        with path.open("rb") as handle:
            if handle.read(4) != _GGUF_MAGIC:
                return meta
            size = handle.seek(0, 2)
            if size < 24:
                return meta
            handle.seek(0)
            with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                if mm[:4] != _GGUF_MAGIC:
                    return meta
                _version, _tensor_count, kv_count = struct.unpack_from(
                    _GGUF_HEADER_FMT, mm, 4
                )
                offset = 24
                for _ in range(kv_count):
                    if offset + 12 > size:
                        break
                    (key_len,) = struct.unpack_from("<Q", mm, offset)
                    offset += 8
                    if offset + key_len + 4 > size:
                        break
                    key = mm[offset : offset + key_len]
                    offset += key_len
                    (value_type,) = struct.unpack_from("<I", mm, offset)
                    offset += 4

                    if key == _GGUF_KEY_ARCH and value_type == _GGUF_TYPE_STRING:
                        (value_len,) = struct.unpack_from("<Q", mm, offset)
                        start = offset + 8
                        end = start + value_len
                        if end <= size:
                            meta.architecture = mm[start:end].decode(
                                "utf-8", errors="replace"
                            )
                        offset = end
                    elif (
                        key.endswith(_GGUF_SUFFIX_CTX_LEN)
                        and value_type == _GGUF_TYPE_U32
                    ):
                        meta.context_length = int(
                            struct.unpack_from("<I", mm, offset)[0]
                        )
                        offset += 4
                    elif (
                        key.endswith(_GGUF_SUFFIX_BLOCK_CNT)
                        and value_type == _GGUF_TYPE_U32
                    ):
                        meta.block_count = int(struct.unpack_from("<I", mm, offset)[0])
                        offset += 4
                    elif (
                        key.endswith(_GGUF_SUFFIX_SLIDING_WIN)
                        and value_type == _GGUF_TYPE_U32
                    ):
                        meta.has_sliding_window = (
                            int(struct.unpack_from("<I", mm, offset)[0]) > 0
                            or meta.has_sliding_window
                        )
                        offset += 4
                    else:
                        if key.endswith(_GGUF_SUFFIX_SLIDING_PAT):
                            meta.has_sliding_window = True
                        offset = _skip_gguf_mmap_value(mm, offset, value_type)
                    if offset > size:
                        break
    except (OSError, ValueError, struct.error):
        return _GGUFMetadata()
    return meta


def _gguf_metadata(model_path: str) -> _GGUFMetadata:
    try:
        path = resolve_gguf_file(model_path)
    except ValueError:
        return _GGUFMetadata()

    cache_key = _gguf_cache_key(path)
    if cache_key is None:
        return _read_gguf_metadata(path)
    cached = _gguf_metadata_cache.get(cache_key)
    if cached is not None:
        _gguf_metadata_cache.move_to_end(cache_key)
        return cached
    meta = _read_gguf_metadata(path)
    _gguf_metadata_cache[cache_key] = meta
    _gguf_metadata_cache.move_to_end(cache_key)
    while len(_gguf_metadata_cache) > _GGUF_METADATA_CACHE_MAX_ENTRIES:
        _gguf_metadata_cache.popitem(last=False)
    return meta


def gguf_architecture(model_path: str) -> str | None:
    """Read ``general.architecture`` from a GGUF file when available."""
    return _gguf_metadata(model_path).architecture


def gguf_context_length(model_path: str) -> int | None:
    """Read training context length from GGUF metadata (e.g. llama.context_length)."""
    return _gguf_metadata(model_path).context_length


def gguf_uses_sliding_window_attention(model_path: str) -> bool:
    """True when GGUF metadata indicates sliding-window attention layers."""
    return _gguf_metadata(model_path).has_sliding_window


def gguf_block_count(model_path: str) -> int | None:
    """Read transformer block count from GGUF metadata when available."""
    return _gguf_metadata(model_path).block_count


def gguf_total_layers(model_path: str | Path) -> int:
    """Block count with a conservative fallback when GGUF metadata is missing."""
    return gguf_block_count(str(model_path)) or 64


def gguf_is_supported_by_llamacpp(model_path: str) -> bool:
    architecture = gguf_architecture(model_path)
    return architecture not in _UNSUPPORTED_GGUF_ARCHITECTURES


def recommend_backend(
    *, model_path: str, model_format: str | None = None
) -> BackendName:
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


def available_backends(
    *, model_path: str, model_format: str | None = None
) -> list[BackendName]:
    """Backends that can serve this inventory model."""
    fmt = (model_format or "").lower()
    path = Path(model_path)
    if fmt == "gguf" and not gguf_is_supported_by_llamacpp(model_path):
        return []
    if fmt == "gguf" or _is_gguf_path(model_path) or path.suffix.lower() == ".gguf":
        return [BACKEND_LLAMASWAP, BACKEND_LLAMACPP]
    if fmt in {"safetensors", "bin"} or path.is_dir():
        preferred = recommend_backend(model_path=model_path, model_format=model_format)
        return list(dict.fromkeys([preferred, BACKEND_TORCH, BACKEND_MLX]))
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
    if choice in {BACKEND_LLAMACPP, BACKEND_LLAMASWAP, BACKEND_MLX, BACKEND_TORCH}:
        return choice
    raise ValueError(f"Unsupported inference backend: {requested}")


def prepare_model_path(model_path: str, backend: BackendName) -> str:
    """Normalize model path (e.g. pick a GGUF file inside a directory)."""
    if backend in {BACKEND_LLAMACPP, BACKEND_LLAMASWAP}:
        return str(resolve_gguf_file(model_path))
    path = Path(model_path).expanduser()
    if (
        backend in {BACKEND_TORCH, BACKEND_MLX}
        and path.is_file()
        and path.suffix.lower() in {".safetensors", ".bin"}
        and (path.parent / "config.json").is_file()
    ):
        return str(path.parent.absolute())
    return model_path
