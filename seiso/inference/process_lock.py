"""Backward-compatible aliases for the shared GPU resource lock."""

from __future__ import annotations

from seiso.memory.gpu_resource_lock import (
    acquire_gpu_resource_lock,
    gpu_resource_lock,
    release_gpu_resource_lock,
)

acquire_inference_process_lock = acquire_gpu_resource_lock
release_inference_process_lock = release_gpu_resource_lock
inference_process_lock = gpu_resource_lock

__all__ = [
    "acquire_inference_process_lock",
    "release_inference_process_lock",
    "inference_process_lock",
]
