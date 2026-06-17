"""Shared Hugging Face Hub cache configuration for downloads and training."""

from __future__ import annotations

import os
from pathlib import Path

from seiso.security import resolve_data_dir


def resolve_hf_cache_dir(data_dir: Path | None = None) -> Path:
    """Directory used by huggingface_hub and transformers."""
    if raw := os.environ.get("HUGGINGFACE_HUB_CACHE"):
        return Path(raw).expanduser()
    if raw := os.environ.get("HF_HOME"):
        return Path(raw).expanduser() / "hub"
    root = resolve_data_dir(data_dir)
    return root / "hf_cache"


def configure_hf_hub_cache(data_dir: Path | None = None) -> Path:
    """Point HF libraries at Seiso's shared cache when not already configured."""
    cache = resolve_hf_cache_dir(data_dir)
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache))
    return cache
