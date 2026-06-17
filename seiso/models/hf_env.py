"""Shared Hugging Face Hub cache configuration for downloads and training."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from seiso.security import resolve_data_dir


def resolve_hf_cache_dir(data_dir: Path | None = None) -> Path:
    """Directory used by huggingface_hub and transformers."""
    if raw := os.environ.get("HUGGINGFACE_HUB_CACHE"):
        return Path(raw).expanduser()
    if raw := os.environ.get("HF_HOME"):
        return Path(raw).expanduser() / "hub"
    root = resolve_data_dir(data_dir)
    return root / "hf_cache"


def _xet_available() -> bool:
    try:
        import hf_xet  # noqa: F401

        return True
    except ImportError:
        return False


def hf_transfer_stack() -> dict[str, Any]:
    """
    Report the active Hub transfer backend.

    Modern huggingface_hub uses hf_xet (Rust/Xet chunk storage) for fast parallel
    downloads. The legacy hf_transfer package is no longer used.
    """
    xet_available = _xet_available()
    xet_version: str | None = None
    if xet_available:
        import hf_xet

        xet_version = getattr(hf_xet, "__version__", None)

    high_perf = os.environ.get("HF_XET_HIGH_PERFORMANCE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    num_threads = os.environ.get("HF_HUB_NUM_THREADS", "8").strip()
    download_timeout = os.environ.get("HF_HUB_DOWNLOAD_TIMEOUT", "300").strip()
    hints: list[str] = []
    if not xet_available:
        hints.append("Install hf-xet for faster parallel downloads: pip install hf-xet")
    elif not high_perf:
        hints.append("Set HF_XET_HIGH_PERFORMANCE=1 for max throughput on fast links")
    return {
        "backend": "hf_xet" if xet_available else "http",
        "xet_available": xet_available,
        "xet_version": xet_version,
        "high_performance": high_perf,
        "num_threads": num_threads,
        "download_timeout_s": download_timeout,
        "hints": hints,
        "hint": hints[0] if hints else None,
    }


def configure_hf_hub_cache(data_dir: Path | None = None) -> Path:
    """
    Point HF libraries at Seiso's shared cache and apply standard transfer tuning.

    Uses huggingface_hub defaults plus:
    - Longer timeouts for multi-GB model files on slow links
    - Parallel snapshot shard downloads (HF_HUB_NUM_THREADS)
    - hf_xet when installed (Rust-backed chunk transfers — no custom Rust needed)
    """
    cache = resolve_hf_cache_dir(data_dir)
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache))
    # Generous defaults for large GGUF / safetensors downloads
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
    # More parallel shard downloads when hf-xet is available
    default_threads = "12" if _xet_available() else "8"
    os.environ.setdefault("HF_HUB_NUM_THREADS", default_threads)
    return cache
