"""Inference backend selection for local models."""

from __future__ import annotations

import mmap
import platform
import re
import struct
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from seiso.compat import StrEnum
from seiso.env import env_bool
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
    "llamaswap": "GGUF sidecar",
    "mlx": "MLX",
    "torch": "PyTorch",
    "router": "Smart Router",
    "auto": "Auto",
}


def resolve_backend_label(
    backend: str,
    *,
    sidecar_engine: str | None = None,
) -> str:
    """Human label for a backend; sidecar label reflects active engine when known."""
    if backend == BACKEND_LLAMASWAP:
        if sidecar_engine == "ollama":
            return "Ollama sidecar"
        if sidecar_engine == "llamacpp":
            return "llama-swap"
        return BACKEND_LABELS[BACKEND_LLAMASWAP]
    return BACKEND_LABELS.get(backend, backend)
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
_GGUF_UINT_FMT = {4: "<I", 5: "<i", 10: "<Q", 11: "<q"}
_GGUF_MAGIC = b"GGUF"
_GGUF_HEADER_FMT = "<IQQ"
_GGUF_KEY_ARCH = b"general.architecture"
_GGUF_SUFFIX_CTX_LEN = b".context_length"
_GGUF_SUFFIX_BLOCK_CNT = b".block_count"
_GGUF_SUFFIX_SLIDING_WIN = b".attention.sliding_window"
_GGUF_SUFFIX_SLIDING_PAT = b".attention.sliding_window_pattern"
_GGUF_SUFFIX_HEAD_CNT = b".attention.head_count"
_GGUF_SUFFIX_HEAD_CNT_KV = b".attention.head_count_kv"
_GGUF_SUFFIX_KEY_LEN = b".attention.key_length"
_GGUF_SUFFIX_VAL_LEN = b".attention.value_length"
_GGUF_SUFFIX_EMBED_LEN = b".embedding_length"
_GGUF_SUFFIX_EXPERT_CNT = b".expert_count"
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
    sliding_window: int | None = None
    swa_layer_fraction: float | None = None
    head_count: int | None = None
    head_count_kv: float | None = None
    key_length: int | None = None
    value_length: int | None = None
    embedding_length: int | None = None
    expert_count: int | None = None


def is_dflash_draft(model_path: str) -> bool:
    """Detect if a GGUF path is a dflash draft model for speculative decoding."""
    if not _is_gguf_path(model_path):
        return False
    arch = gguf_architecture(model_path)
    if arch and "dflash" in arch.lower():
        return True
    # Name must explicitly say dflash; bare "-draft" matches too many community models.
    return "dflash" in Path(model_path).name.lower()


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


def _unpack_gguf_uint(
    mm: mmap.mmap, offset: int, value_type: int
) -> tuple[int | None, int]:
    """Read a scalar integer GGUF value, returning (value, new_offset)."""
    fmt = _GGUF_UINT_FMT.get(value_type)
    if fmt is None:
        return None, _skip_gguf_mmap_value(mm, offset, value_type)
    (value,) = struct.unpack_from(fmt, mm, offset)
    return int(value), offset + _GGUF_VALUE_SIZE[value_type]


def _mean_gguf_uint_array(
    mm: mmap.mmap, offset: int, value_type: int
) -> tuple[float | None, int]:
    """Mean of an integer GGUF array value (e.g. per-layer head_count_kv)."""
    if value_type != _GGUF_TYPE_ARRAY:
        return None, _skip_gguf_mmap_value(mm, offset, value_type)
    item_type, count = struct.unpack_from("<IQ", mm, offset)
    fmt = _GGUF_UINT_FMT.get(item_type)
    if fmt is None or count == 0:
        return None, _skip_gguf_mmap_value(mm, offset, value_type)
    item_size = _GGUF_VALUE_SIZE[item_type]
    start = offset + 12
    values = [
        int(struct.unpack_from(fmt, mm, start + i * item_size)[0])
        for i in range(count)
    ]
    return sum(values) / len(values), start + item_size * count


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
                    elif key.endswith(_GGUF_SUFFIX_CTX_LEN):
                        value, offset = _unpack_gguf_uint(mm, offset, value_type)
                        if value is not None:
                            meta.context_length = value
                    elif key.endswith(_GGUF_SUFFIX_BLOCK_CNT):
                        value, offset = _unpack_gguf_uint(mm, offset, value_type)
                        if value is not None:
                            meta.block_count = value
                    elif key.endswith(_GGUF_SUFFIX_SLIDING_WIN):
                        value, offset = _unpack_gguf_uint(mm, offset, value_type)
                        if value is not None:
                            if value > 0:
                                meta.sliding_window = value
                            meta.has_sliding_window = (
                                value > 0 or meta.has_sliding_window
                            )
                    elif key.endswith(_GGUF_SUFFIX_HEAD_CNT_KV):
                        # Per-layer arrays appear on mixed-attention models
                        # (e.g. gemma); the mean is exact for total KV size.
                        if value_type == _GGUF_TYPE_ARRAY:
                            mean, offset = _mean_gguf_uint_array(
                                mm, offset, value_type
                            )
                            if mean is not None:
                                meta.head_count_kv = mean
                        else:
                            value, offset = _unpack_gguf_uint(mm, offset, value_type)
                            if value is not None:
                                meta.head_count_kv = float(value)
                    elif key.endswith(_GGUF_SUFFIX_HEAD_CNT):
                        value, offset = _unpack_gguf_uint(mm, offset, value_type)
                        if value is not None:
                            meta.head_count = value
                    elif key.endswith(_GGUF_SUFFIX_KEY_LEN):
                        value, offset = _unpack_gguf_uint(mm, offset, value_type)
                        if value is not None:
                            meta.key_length = value
                    elif key.endswith(_GGUF_SUFFIX_VAL_LEN):
                        value, offset = _unpack_gguf_uint(mm, offset, value_type)
                        if value is not None:
                            meta.value_length = value
                    elif key.endswith(_GGUF_SUFFIX_EMBED_LEN):
                        value, offset = _unpack_gguf_uint(mm, offset, value_type)
                        if value is not None:
                            meta.embedding_length = value
                    elif key.endswith(_GGUF_SUFFIX_EXPERT_CNT):
                        value, offset = _unpack_gguf_uint(mm, offset, value_type)
                        if value is not None and value > 0:
                            meta.expert_count = value
                    else:
                        if key.endswith(_GGUF_SUFFIX_SLIDING_PAT):
                            meta.has_sliding_window = True
                            if value_type == _GGUF_TYPE_ARRAY:
                                mean, offset = _mean_gguf_uint_array(
                                    mm, offset, value_type
                                )
                                if mean is not None:
                                    meta.swa_layer_fraction = mean
                            else:
                                value, offset = _unpack_gguf_uint(
                                    mm, offset, value_type
                                )
                                if value is not None:
                                    meta.swa_layer_fraction = float(value)
                                else:
                                    offset = _skip_gguf_mmap_value(
                                        mm, offset, value_type
                                    )
                            continue
                        offset = _skip_gguf_mmap_value(mm, offset, value_type)
                    if offset > size:
                        break
    except OSError:
        return _GGUFMetadata()
    except (ValueError, struct.error):
        # Some community GGUFs contain newer metadata value types before the
        # tensor table. Keep fields parsed before the unknown entry so safety
        # policy does not silently fall back to an unsafe dense default.
        return meta
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


def gguf_sliding_window(model_path: str) -> int | None:
    """Sliding-window span from GGUF metadata (Gemma 3/4, etc.), when present."""
    return _gguf_metadata(model_path).sliding_window


def gguf_swa_layer_fraction(model_path: str) -> float | None:
    """Fraction of transformer blocks using sliding-window (local) attention."""
    return _gguf_metadata(model_path).swa_layer_fraction


def gguf_is_moe(model_path: str) -> bool:
    """True when GGUF metadata indicates a mixture-of-experts architecture."""
    meta = _gguf_metadata(model_path)
    if meta.expert_count and meta.expert_count > 1:
        return True
    arch = (meta.architecture or "").lower()
    return "moe" in arch


def gguf_block_count(model_path: str) -> int | None:
    """Read transformer block count from GGUF metadata when available."""
    return _gguf_metadata(model_path).block_count


def gguf_total_layers(model_path: str | Path) -> int:
    """Block count with a conservative fallback when GGUF metadata is missing."""
    return gguf_block_count(str(model_path)) or 64


def gguf_kv_bytes_per_token(model_path: str) -> int | None:
    """Exact fp16 KV-cache bytes per context token from GGUF metadata.

    Uses attention geometry (GQA-aware) instead of parameter-count heuristics:
    ``block_count * head_count_kv * (key_length + value_length) * 2 bytes``.
    Returns ``None`` when the metadata is insufficient.
    """
    meta = _gguf_metadata(model_path)
    if not meta.block_count:
        return None

    kv_heads = meta.head_count_kv
    if kv_heads is None:
        kv_heads = float(meta.head_count) if meta.head_count else None

    key_length = meta.key_length
    value_length = meta.value_length
    if (key_length is None or value_length is None) and (
        meta.embedding_length and meta.head_count
    ):
        head_dim = meta.embedding_length // meta.head_count
        key_length = key_length if key_length is not None else head_dim
        value_length = value_length if value_length is not None else head_dim

    if not kv_heads or not key_length or not value_length:
        return None

    bytes_per_token = meta.block_count * kv_heads * (key_length + value_length) * 2
    return int(bytes_per_token) or None


def gguf_is_supported_by_llamacpp(model_path: str) -> bool:
    architecture = gguf_architecture(model_path)
    return architecture not in _UNSUPPORTED_GGUF_ARCHITECTURES


def _is_gguf_model(model_path: str, model_format: str | None = None) -> bool:
    fmt = (model_format or "").lower()
    path = Path(model_path)
    if fmt and fmt != "gguf":
        # Inventory metadata is authoritative. Mixed export/cache directories can
        # contain helper GGUFs next to safetensors; do not reroute those to GGUF.
        return False
    return fmt == "gguf" or _is_gguf_path(model_path) or path.suffix.lower() == ".gguf"


def _native_linux_requires_isolated_gguf() -> bool:
    """True when GGUF chat must run out-of-process (Ollama/llama-swap).

    Fails **closed** on bare-metal Linux: a detection error (e.g. a broken CUDA
    runtime making the torch probe raise) must never re-enable the in-process
    ``llama.cpp`` path, because a CUDA fault there would crash Forge.
    """
    if env_bool("SEISO_LLAMA_ALLOW_INPROCESS_NATIVE_LINUX", False):
        return False

    # Primary signal — also governs WSL via SEISO_NVIDIA_WSL_ACK.
    try:
        from seiso.platform import use_linux_nvidia_inference_guards

        if use_linux_nvidia_inference_guards():
            return True
    except Exception:
        # Detection hiccup (e.g. torch CUDA probe error); fall through to the
        # torch-free nvidia-smi signal below instead of failing open.
        pass

    # Only bare-metal Linux uses the nvidia-smi fallback. Non-Linux hosts and
    # WSL keep the result above (WSL requires an explicit ack via the guards).
    if platform.system() != "Linux":
        return False
    try:
        from seiso.platform import detect_wsl2

        if detect_wsl2():
            return False
    except Exception:
        pass

    try:
        from seiso.security.nvidia_boundary import nvidia_smi_visible

        return nvidia_smi_visible()
    except Exception:
        # Cannot rule out an NVIDIA GPU on bare-metal Linux; fail closed so a
        # llama.cpp CUDA fault cannot crash Forge in-process.
        return True


def _llamaswap_unavailable_error(reason: str | None = None) -> RuntimeError:
    detail = f" {reason}" if reason else ""
    return RuntimeError(
        "Native Linux NVIDIA GGUF chat requires an isolated backend to prevent "
        "llama.cpp CUDA crashes from killing Forge. Start llama-swap/Ollama or set "
        "SEISO_LLAMA_ALLOW_INPROCESS_NATIVE_LINUX=1 to explicitly accept in-process "
        f"llama.cpp risk.{detail}"
    )


def _assert_llamaswap_available() -> None:
    from seiso.inference.llamaswap import llamaswap_status

    status = llamaswap_status()
    if not status.available:
        raise _llamaswap_unavailable_error(status.reason)


def recommend_backend(
    *, model_path: str, model_format: str | None = None
) -> BackendName:
    """Pick the single local inference engine from model format and host policy."""
    fmt = (model_format or "").lower()
    path = Path(model_path)
    if _is_gguf_model(model_path, model_format):
        return (
            BACKEND_LLAMASWAP
            if _native_linux_requires_isolated_gguf()
            else BACKEND_LLAMACPP
        )
    if fmt in {"safetensors", "bin"} or path.is_dir():
        if platform.system() == "Darwin" and detect_backend() == Backend.MLX:
            return BACKEND_MLX
        return BACKEND_TORCH
    return BACKEND_TORCH


def available_backends(
    *, model_path: str, model_format: str | None = None
) -> list[BackendName]:
    """Local backends exposed for this model.

    Keep this deterministic. Complex model selection belongs to the optional
    external router, not the local backend resolver.
    """
    fmt = (model_format or "").lower()
    if fmt == "gguf" and not gguf_is_supported_by_llamacpp(model_path):
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
    is_gguf = _is_gguf_model(model_path, model_format)
    if choice == BACKEND_AUTO:
        if is_gguf and _native_linux_requires_isolated_gguf():
            _assert_llamaswap_available()
            return BACKEND_LLAMASWAP
        return recommend_backend(model_path=model_path, model_format=model_format)
    if (
        choice == BACKEND_LLAMACPP
        and is_gguf
        and _native_linux_requires_isolated_gguf()
    ):
        raise _llamaswap_unavailable_error(
            "The requested backend was llamacpp; use llamaswap or set "
            "SEISO_LLAMA_ALLOW_INPROCESS_NATIVE_LINUX=1."
        )

    if is_gguf:
        if choice in {BACKEND_LLAMACPP, BACKEND_LLAMASWAP}:
            if choice == BACKEND_LLAMASWAP:
                _assert_llamaswap_available()
            return choice
        if choice in {BACKEND_MLX, BACKEND_TORCH}:
            raise ValueError(f"Backend {choice!r} cannot load GGUF models")
    else:
        if choice in {BACKEND_MLX, BACKEND_TORCH}:
            return choice
        if choice in {BACKEND_LLAMACPP, BACKEND_LLAMASWAP}:
            raise ValueError(f"Backend {choice!r} requires a GGUF model")

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
