"""Shared environment-variable parsing helpers."""

from __future__ import annotations

import os


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def env_str(name: str, default: str) -> str:
    raw = os.environ.get(name, "").strip()
    return raw if raw else default


def configure_transformers_env() -> None:
    """Env vars that must be set before ``transformers`` / ``trl`` import."""
    # kernels>=0.13 + transformers 5.12 crash at import unless hub kernels are off.
    os.environ.setdefault("USE_HUB_KERNELS", "NO")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
