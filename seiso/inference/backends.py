"""Inference backend selection for local models."""

from __future__ import annotations

import mmap
import os
import re
import struct
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from seiso.compat import StrEnum
from seiso.models.loader import Backend, detect_backend

BackendName = str


# ── Enum & constants ───────────────────────────────────────────

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


# ── GGUF internals (byte-level, zero-decode) ───────────────────

_GGUF_SHARD_RE = re.compile(
    r"^(?P<prefix>.+)-(?P<index>\d{5})-of-(?P<total>\d{5})\.gguf$", re.I
)

# Pre-compiled byte keys/suffixes — compared directly against mmap slices
_GGUF_KEY_ARCH = b"general.architecture"
_GGUF_SUFFIX_CTX_LEN = b".context_length"
_GGUF_SUFFIX_BLOCK_CNT = b".block_count"
_GGUF_SUFFIX_SLIDING_WIN = b".attention.sliding_window"
_GGUF_SUFFIX_SLIDING_PAT = b".attention.sliding_window_pattern"

# GGUF value type constants
_GGUF_TYPE_U32 = 4
_GGUF_TYPE_STRING = 8
_GGUF_TYPE_ARRAY = 9

# O(1) value-size lookup: index 0–7 → size, types 10–12 all map to 8
_VALUE_SIZE = (1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8)

_GGUF_HEADER_FMT = "<IQQ"
_GGUF_MAGIC = b"GGUF"


@dataclass(slots=True)
class _GGUFMetadata:
    """Single-pass GGUF metadata cache."""

    architecture: str | None = None
    context_length: int | None = None
    block_count: int | None = None
    has_sliding_window: bool = False


# ── Path helpers ───────────────────────────────────────────────

def _is_gguf_file(path: Path) -> bool:
    if path.suffix.lower() == ".gguf":
        return True
    try:
        with path.open("rb") as handle:
            return handle.read(4) == _GGUF_MAGIC
    except OSError:
        return False


def _is_gguf_path(model_path: str) -> bool:
    path = Path(model_path)
    if path.is_file():
        return _is_gguf_file(path)
    if path.is_dir():
        return any(path.glob("*.gguf"))
    return False


def resolve_gguf_file(model_path: str) -> Path:
    """Pick a single GGUF file from a path or directory."""
    path = Path(model_path).expanduser()

    if path.is_file():
        if _is_gguf_file(path):
            return path.absolute()
        raise ValueError(f"Not a GGUF file: {path}")

    if path.is_dir():
        candidates = sorted(path.glob("*.gguf"))
        for candidate in candidates:
            match = _GGUF_SHARD_RE.match(candidate.name)
            if match and match.group("index") == "00001":
                return candidate.absolute()
        if candidates:
            return max(candidates, key=lambda p: p.stat().st_size).absolute()

    raise ValueError(f"No GGUF file found at {model_path}")


# ── Speculative Decoding Support ───────────────────────────────

def is_draft_model(model_path: str) -> bool:
    """Generalized check if a model is intended for speculative drafting.

    Uses filename heuristics to avoid loading metadata solely for this check.
    """
    name = Path(model_path).name.lower()
    # Common naming conventions for draft models
    return "draft" in name or "speculative" in name


# ── mmap single-pass GGUF reader (fully inlined hot path) ──────

def _read_gguf_metadata_all(path: Path) -> _GGUFMetadata:
    """Read all metadata from a GGUF file in one pass via mmap.

    Zero-copy memory-mapped I/O, byte-level key matching, fully inlined
    value-skipping — no function calls in the KV loop.
    """
    meta = _GGUFMetadata()

    try:
        with open(path, "rb") as f:
            file_size = os.fstat(f.fileno()).st_size
            if file_size < 4 + struct.calcsize(_GGUF_HEADER_FMT):
                return meta

            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
                if m[0:4] != _GGUF_MAGIC:
                    return meta

                _, _, kv_count = struct.unpack_from(_GGUF_HEADER_FMT, m, 4)
                offset = 24  # 4 (magic) + 20 (header)

                for _ in range(kv_count):
                    # ── Read key ──
                    key_len = struct.unpack_from("<Q", m, offset)[0]
                    offset += 8
                    if offset + key_len > file_size:
                        break
                    key_bytes = m[offset : offset + key_len]
                    offset += key_len

                    # ── Read value type ──
                    if offset + 4 > file_size:
                        break
                    value_type = struct.unpack_from("<I", m, offset)[0]
                    offset += 4

                    # ── Match & collect (inline — no function calls) ──

                    # general.architecture
                    if key_bytes == _GGUF_KEY_ARCH and value_type == _GGUF_TYPE_STRING:
                        val_len = struct.unpack_from("<Q", m, offset)[0]
                        meta.architecture = m[offset + 8 : offset + 8 + val_len].decode(
                            "utf-8", errors="replace"
                        )
                        offset += 8 + val_len
                        continue

                    # *.context_length (u32)
                    if key_bytes.endswith(_GGUF_SUFFIX_CTX_LEN) and value_type == _GGUF_TYPE_U32:
                        meta.context_length = struct.unpack_from("<I", m, offset)[0]
                        offset += 4
                        continue

                    # *.block_count (u32)
                    if key_bytes.endswith(_GGUF_SUFFIX_BLOCK_CNT) and value_type == _GGUF_TYPE_U32:
                        meta.block_count = struct.unpack_from("<I", m, offset)[0]
                        offset += 4
                        continue

                    # *.attention.sliding_window (u32 > 0)
                    if key_bytes.endswith(_GGUF_SUFFIX_SLIDING_WIN) and value_type == _GGUF_TYPE_U32:
                        if struct.unpack_from("<I", m, offset)[0] > 0:
                            meta.has_sliding_window = True
                        offset += 4
                        continue

                    # *.attention.sliding_window_pattern (presence check)
                    if key_bytes.endswith(_GGUF_SUFFIX_SLIDING_PAT):
                        meta.has_sliding_window = True

                    # ── Skip non-target value (inline) ──
                    if value_type == _GGUF_TYPE_STRING:
                        val_len = struct.unpack_from("<Q", m, offset)[0]
                        offset += 8 + val_len

                    elif value_type == _GGUF_TYPE_ARRAY:
                        item_type, count = struct.unpack_from("<IQ", m, offset)
                        offset += 12
                        if item_type == _GGUF_TYPE_STRING:
                            for _ in range(count):
                                vlen = struct.unpack_from("<Q", m, offset)[0]
                                offset += 8 + vlen
                        else:
                            sz = (
                                _VALUE_SIZE[item_type]
                                if item_type < len(_VALUE_SIZE)
                                else 0
                            )
                            offset += sz * count

                    else:
                        sz = _VALUE_SIZE[value_type] if value_type < len(_VALUE_SIZE) else 8
                        offset += sz

    except (OSError, ValueError, struct.error, mmap.error):
        return _GGUFMetadata()

    return meta


# ── Caching (bounded LRU) ──────────────────────────────────────

@lru_cache(maxsize=128)
def _resolve_gguf_file_cached(model_path: str) -> Path:
    """Cached GGUF file resolution."""
    return resolve_gguf_file(model_path)


@lru_cache(maxsize=128)
def _get_metadata_cached(path_str: str, mtime: float, size: int) -> _GGUFMetadata:
    """Cached single-pass metadata read keyed by (path, mtime, size)."""
    return _read_gguf_metadata_all(Path(path_str))


def clear_gguf_caches() -> None:
    """Reset all GGUF caches (for tests)."""
    _resolve_gguf_file_cached.cache_clear()
    _get_metadata_cached.cache_clear()


def _get_metadata(model_path: str) -> _GGUFMetadata:
    """Get cached or freshly-read GGUF metadata."""
    try:
        path = _resolve_gguf_file_cached(model_path)
    except ValueError:
        return _GGUFMetadata()

    try:
        stat = path.stat()
        return _get_metadata_cached(str(path), stat.st_mtime, stat.st_size)
    except OSError:
        return _GGUFMetadata()


# ── Public GGUF metadata readers ───────────────────────────────

def gguf_architecture(model_path: str) -> str | None:
    """Read ``general.architecture`` from a GGUF file when available."""
    return _get_metadata(model_path).architecture


def gguf_context_length(model_path: str) -> int | None:
    """Read training context length from GGUF metadata (e.g. llama.context_length)."""
    return _get_metadata(model_path).context_length


def gguf_uses_sliding_window_attention(model_path: str) -> bool:
    """True when GGUF metadata indicates sliding-window attention layers."""
    return _get_metadata(model_path).has_sliding_window


def gguf_block_count(model_path: str) -> int | None:
    """Read transformer block count from GGUF metadata when available."""
    return _get_metadata(model_path).block_count


# ── Backend resolution ─────────────────────────────────────────

def gguf_is_supported_by_llamacpp(_model_path: str) -> bool:
    """All GGUF models are supported by llama.cpp."""
    return True


def recommend_backend(*, model_path: str, model_format: str | None = None) -> BackendName:
    """Pick the default local inference engine from model path/format.

    Note: Draft models (for speculative decoding) are typically small GGUFs,
    so they correctly resolve to LLAMACPP here automatically.
    """
    fmt = (model_format or "").lower()

    if fmt == "gguf" or _is_gguf_path(model_path):
        return BACKEND_LLAMACPP

    path = Path(model_path)
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
        return str(_resolve_gguf_file_cached(model_path))
    return model_path
