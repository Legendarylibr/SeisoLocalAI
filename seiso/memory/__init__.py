"""Memory safety utilities — OOM prevention across inference, training, and RL."""

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
    run_with_oom_retry,
    sanitize_inference_payload,
    training_pin_memory,
)

__all__ = [
    "MemoryLoadBlockedError",
    "apply_rl_memory_guards",
    "apply_training_memory_guards",
    "assess_path_memory_fit",
    "build_hf_max_memory",
    "clamp_llama_cache_mb",
    "clamp_llama_load_kwargs",
    "ensure_load_fits",
    "headroom_mb",
    "is_oom_error",
    "release_cached_memory",
    "run_with_oom_retry",
    "sanitize_inference_payload",
    "training_pin_memory",
]
