"""Capability and memory-aware KV-cache policy for native Linux inference."""

from __future__ import annotations

import platform
from dataclasses import asdict, dataclass
from typing import Any

from seiso.env import env_bool, env_int, env_str

_CACHE_IMPLEMENTATIONS = {"dynamic", "static", "offloaded", "quantized"}


@dataclass(frozen=True, slots=True)
class KVCachePolicy:
    """Resolved per-request cache policy with conservative defaults."""

    cache_implementation: str
    prefill_chunk_size: int
    prefix_cache: bool
    compile_decode: bool
    kv_bits: int
    estimated_cache_mb: int
    headroom_mb: int
    fallback_reason: str | None = None

    @property
    def manual_stream_compatible(self) -> bool:
        return self.cache_implementation == "dynamic" and self.kv_bits >= 16

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SidecarKVPolicy:
    """Only options positively advertised by an isolated sidecar."""

    prompt_reuse: bool = True
    num_keep: int | None = None
    cache_prompt: bool | None = None


def resolve_sidecar_kv_policy(payload: dict[str, Any], *, engine: str) -> SidecarKVPolicy:
    """Negotiate optional sidecar controls without guessing API support."""
    raw = payload.get("sidecar_capabilities")
    if isinstance(raw, str):
        capabilities = {item.strip().lower() for item in raw.split(",") if item.strip()}
    elif isinstance(raw, (list, tuple, set)):
        capabilities = {str(item).strip().lower() for item in raw}
    else:
        capabilities = set()

    if engine == "ollama":
        configured = env_int("SEISO_OLLAMA_NUM_KEEP", -1)
        num_keep = (
            max(0, configured)
            if configured >= 0
            else (
                max(0, int(payload.get("sidecar_num_keep", 0)))
                if "num_keep" in capabilities
                else None
            )
        )
        return SidecarKVPolicy(num_keep=num_keep)
    return SidecarKVPolicy(cache_prompt=True if "cache_prompt" in capabilities else None)


def estimate_torch_kv_cache_mb(model: Any, tokens: int, *, bits: int = 16) -> int:
    """Estimate decoder KV memory from common Transformers config fields."""
    config = getattr(model, "config", None)
    if config is None or tokens <= 0:
        return 0
    layers = int(getattr(config, "num_hidden_layers", 0) or getattr(config, "n_layer", 0) or 0)
    heads = int(
        getattr(config, "num_key_value_heads", 0)
        or getattr(config, "num_attention_heads", 0)
        or getattr(config, "n_head", 0)
        or 0
    )
    hidden = int(getattr(config, "hidden_size", 0) or getattr(config, "n_embd", 0) or 0)
    attention_heads = int(
        getattr(config, "num_attention_heads", 0) or getattr(config, "n_head", 0) or heads or 1
    )
    if layers <= 0 or heads <= 0 or hidden <= 0:
        return 0
    head_dim = max(1, hidden // attention_heads)
    cache_bytes = 2 * layers * heads * head_dim * int(tokens) * max(2, int(bits)) / 8
    return max(1, int(cache_bytes / (1024**2) * 1.10))


def _requested_cache_implementation(payload: dict[str, Any]) -> str:
    policy = payload.get("kv_policy")
    raw = policy.get("cache_implementation") if isinstance(policy, dict) else None
    if raw is None:
        raw = payload.get("cache_implementation")
    if raw is None:
        raw = env_str("SEISO_TORCH_CACHE_IMPLEMENTATION", "dynamic")
    value = str(raw).strip().lower()
    return value if value in _CACHE_IMPLEMENTATIONS else "dynamic"


def resolve_kv_cache_policy(
    payload: dict[str, Any],
    *,
    model: Any,
    input_tokens: int,
    max_tokens: int,
    free_mb: int,
) -> KVCachePolicy:
    """Resolve a safe policy; unsupported/high-risk modes fall back to dynamic."""
    total_tokens = max(1, int(input_tokens) + max(1, int(max_tokens)))
    requested = _requested_cache_implementation(payload)
    reason: str | None = None
    kv_bits = 16
    if requested == "quantized":
        explicit = payload.get("cache_implementation") == "quantized" or env_bool(
            "SEISO_TORCH_QUANTIZED_KV", False
        )
        if explicit:
            kv_bits = max(2, min(env_int("SEISO_TORCH_KV_BITS", 8), 8))
        else:
            requested = "dynamic"
            reason = "quantized KV requires explicit opt-in"

    estimated = estimate_torch_kv_cache_mb(model, total_tokens, bits=kv_bits)
    usable = max(0, int(free_mb) - env_int("SEISO_TORCH_KV_HEADROOM_MB", 768))
    if estimated > 0 and usable > 0 and estimated > usable:
        if requested in {"static", "offloaded"}:
            requested = "dynamic"
            reason = "requested cache exceeds reserved GPU headroom"
        elif requested == "quantized":
            requested = "dynamic"
            kv_bits = 16
            estimated = estimate_torch_kv_cache_mb(model, total_tokens, bits=kv_bits)
            reason = "quantized cache exceeds reserved GPU headroom"

    threshold = max(256, env_int("SEISO_TORCH_PREFILL_CHUNK_THRESHOLD", 2048))
    configured_chunk = max(128, env_int("SEISO_TORCH_PREFILL_CHUNK_SIZE", 1024))
    chunk = configured_chunk if input_tokens > threshold else max(1, int(input_tokens))
    if usable > 0 and estimated > usable * 0.75:
        chunk = min(chunk, 512)

    native_linux = platform.system() == "Linux"
    prefix_cache = native_linux and env_bool("SEISO_TORCH_PREFIX_CACHE", True)
    compile_decode = (
        native_linux and requested == "static" and env_bool("SEISO_TORCH_DECODE_GRAPHS", False)
    )
    return KVCachePolicy(
        cache_implementation=requested,
        prefill_chunk_size=max(1, chunk),
        prefix_cache=prefix_cache,
        compile_decode=compile_decode,
        kv_bits=kv_bits,
        estimated_cache_mb=estimated,
        headroom_mb=max(0, int(free_mb)),
        fallback_reason=reason,
    )
