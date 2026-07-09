"""Cross-cutting OOM prevention — headroom probes, cache release, and fallbacks."""

from __future__ import annotations

import logging
from typing import Literal

logger = logging.getLogger(__name__)

# Reserve a slice of free memory for OS / display / other processes.
_DEFAULT_RESERVE_RATIO = 0.03
# Generation + activations overhead on top of weight estimate.
_INFERENCE_OVERHEAD_MB = 256
_TRAINING_OVERHEAD_RATIO = 2.0
# Absolute ceilings — never exceed even on large machines.
_MAX_INFERENCE_TOKENS = 8192
_MAX_LLAMA_CTX = 131072
_MIN_LLAMA_CTX = 2048
_MAX_LLAMA_BATCH = 4096
_MIN_LLAMA_BATCH = 128
# Native Linux NVIDIA: tokens-per-GB batch ceiling and absolute normal-tier cap.
_NATIVE_LINUX_BATCH_TOKENS_PER_GB = 8
_NATIVE_LINUX_MAX_COMPLETION_TOKENS = 768
_NATIVE_LINUX_LOW_HEADROOM_MAX_COMPLETION_TOKENS = 512
_NATIVE_LINUX_MAX_NORMAL_BATCH = 256
_NATIVE_LINUX_COMPACT_BATCH_FLOOR = 64
_NATIVE_LINUX_COMPACT_UBATCH_FLOOR = 32
_NATIVE_LINUX_MINIMAL_BATCH_FLOOR = 32
_NATIVE_LINUX_MINIMAL_UBATCH_FLOOR = 16
# Conservative llama.cpp batch pair when native Linux GPU total is unknown.
_NATIVE_LINUX_UNKNOWN_GPU_BATCH_CAPS: tuple[int, int] = (128, 128)
# Coarse n_ctx buckets — avoid reloading the model every few chat turns.
_LLAMA_CTX_BUCKETS = (
    2048,
    4096,
    8192,
    12288,
    16384,
    24576,
    32768,
    49152,
    65536,
    98304,
    131072,
)
_NATIVE_LINUX_CTX_BUCKETS = (
    2048,
    4096,
    8192,
    12288,
    16384,
    24576,
    32768,
    65536,
    131072,
)
# Post-weight headroom below this on native Linux → clamp batches (prefill crash zone).
_NATIVE_LINUX_PREFILL_CLAMP_MB = 6144
_NATIVE_LINUX_PREFILL_HEADROOM_DROP_RATIO = 0.85
_NATIVE_LINUX_PREFILL_HEADROOM_SHRINK_RATIO = 0.92
_NATIVE_LINUX_PREFILL_RESERVE_PER_256TOK_MB = 256
_NATIVE_LINUX_MMPROJ_RESERVE_MB = 512
_TIGHT_VRAM_FIT_RATIO = 0.65
_NATIVE_LINUX_TIGHT_VRAM_FIT_RATIO = 0.60
# Prefer datasets mmap for any JSONL above this size (was 512; list-load expands heavily).
_MAX_JSONL_LOAD_MB = 32
_MODEL_WEIGHT_VRAM_SUFFIXES = frozenset({".gguf", ".safetensors", ".bin"})

_VRAM_ESTIMATE_CACHE_MAX = 256
_vram_estimate_cache: dict[tuple, int] = {}

LlamaLoadTier = Literal["normal", "compact", "minimal"]

# Load-tier recovery ceilings — absolute fallbacks when GPU total is unknown.
_LOAD_TIER_BATCH_CAPS: dict[LlamaLoadTier, tuple[int, int]] = {
    "normal": (_MAX_LLAMA_BATCH, 1024),
    "compact": (512, 128),
    "minimal": (256, 128),
}

__all__ = [
    "LlamaLoadTier",
    "_DEFAULT_RESERVE_RATIO",
    "_INFERENCE_OVERHEAD_MB",
    "_LLAMA_CTX_BUCKETS",
    "_LOAD_TIER_BATCH_CAPS",
    "_MAX_INFERENCE_TOKENS",
    "_MAX_JSONL_LOAD_MB",
    "_MAX_LLAMA_BATCH",
    "_MAX_LLAMA_CTX",
    "_MIN_LLAMA_BATCH",
    "_MIN_LLAMA_CTX",
    "_NATIVE_LINUX_BATCH_TOKENS_PER_GB",
    "_NATIVE_LINUX_MAX_COMPLETION_TOKENS",
    "_NATIVE_LINUX_LOW_HEADROOM_MAX_COMPLETION_TOKENS",
    "_NATIVE_LINUX_COMPACT_BATCH_FLOOR",
    "_NATIVE_LINUX_COMPACT_UBATCH_FLOOR",
    "_NATIVE_LINUX_MINIMAL_BATCH_FLOOR",
    "_NATIVE_LINUX_MINIMAL_UBATCH_FLOOR",
    "_NATIVE_LINUX_MAX_NORMAL_BATCH",
    "_NATIVE_LINUX_MMPROJ_RESERVE_MB",
    "_NATIVE_LINUX_UNKNOWN_GPU_BATCH_CAPS",
    "_MODEL_WEIGHT_VRAM_SUFFIXES",
    "_NATIVE_LINUX_CTX_BUCKETS",
    "_NATIVE_LINUX_PREFILL_CLAMP_MB",
    "_NATIVE_LINUX_PREFILL_HEADROOM_DROP_RATIO",
    "_NATIVE_LINUX_PREFILL_RESERVE_PER_256TOK_MB",
    "_NATIVE_LINUX_PREFILL_HEADROOM_SHRINK_RATIO",
    "_NATIVE_LINUX_TIGHT_VRAM_FIT_RATIO",
    "_TIGHT_VRAM_FIT_RATIO",
    "_VRAM_ESTIMATE_CACHE_MAX",
    "_vram_estimate_cache",
]
