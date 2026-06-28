"""Hooks for clearing inference runtime caches after installs or env changes."""

from __future__ import annotations

import contextlib
from collections.abc import Callable

_clear_hooks: list[Callable[[], None]] = []


def register_runtime_cache_clear(callback: Callable[[], None]) -> None:
    """Register a cache-clear callback (e.g. Forge hf_connectivity lru_cache)."""
    if callback not in _clear_hooks:
        _clear_hooks.append(callback)


def clear_inference_runtime_caches() -> None:
    """Invoke all registered runtime cache clears."""
    for callback in _clear_hooks:
        with contextlib.suppress(Exception):
            callback()
