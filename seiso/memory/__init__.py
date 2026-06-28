"""Memory safety utilities — OOM prevention across inference, training, and RL."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from seiso.memory.platform_profile import (
        apply_platform_memory_profile,
        memory_profile_label,
    )
    from seiso.memory.protection import (
        MemoryLoadBlockedError,
        apply_rl_memory_guards,
        apply_training_memory_guards,
        assess_path_memory_fit,
        build_hf_max_memory,
        clamp_llama_cache_mb,
        clamp_llama_load_kwargs,
        ensure_load_fits,
        headroom_mb,
        is_oom_error,
        release_cached_memory,
        sanitize_inference_payload,
        training_pin_memory,
    )

__all__ = [
    "MemoryLoadBlockedError",
    "apply_platform_memory_profile",
    "apply_rl_memory_guards",
    "apply_training_memory_guards",
    "assess_path_memory_fit",
    "build_hf_max_memory",
    "clamp_llama_cache_mb",
    "clamp_llama_load_kwargs",
    "ensure_load_fits",
    "headroom_mb",
    "is_oom_error",
    "memory_profile_label",
    "release_cached_memory",
    "sanitize_inference_payload",
    "training_pin_memory",
]


def __getattr__(name: str):
    if name in ("apply_platform_memory_profile", "memory_profile_label"):
        from seiso.memory import platform_profile

        return getattr(platform_profile, name)
    if name in __all__:
        from seiso.memory import protection

        return getattr(protection, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
