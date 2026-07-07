"""Shared guard for standalone GPU-heavy CLI work."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator

from seiso.memory.gpu_resource_lock import (
    acquire_gpu_resource_lock,
    release_gpu_resource_lock,
)

logger = logging.getLogger(__name__)


@contextmanager
def gpu_task(task: str) -> Iterator[None]:
    """Serialize CLI GPU work with Forge/inference and release caches on exit."""
    acquire_gpu_resource_lock()
    try:
        try:
            from seiso.inference.model_pool import get_model_pool

            get_model_pool().prepare_for_load()
        except Exception:
            logger.debug("Failed to unload inference before %s", task, exc_info=True)
            raise
        from seiso.memory.protection import release_cached_memory

        release_cached_memory(sync=True)
        yield
    finally:
        try:
            from seiso.kernels.lifecycle import restore_kernel_patches

            restore_kernel_patches()
        except Exception:
            logger.debug("Failed to restore kernels after %s", task, exc_info=True)
        try:
            from seiso.memory.protection import release_cached_memory

            release_cached_memory(sync=True)
        finally:
            release_gpu_resource_lock()
