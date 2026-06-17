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


def hf_transfer_stack() -> dict[str, Any]:
    """
    Report the active Hub transfer backend.

    Modern huggingface_hub uses hf_xet (Rust/Xet chunk storage) for fast parallel
    downloads. The legacy hf_transfer package is no longer used.
    """
    xet_available = False
    xet_version: str | None = None
    try:
        import hf_xet

        xet_available = True
        xet_version = getattr(hf_xet, "__version__", None)
    except ImportError:
        pass

    high_perf = os.environ.get("HF_XET_HIGH_PERFORMANCE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    return {
        "backend": "hf_xet" if xet_available else "http",
        "xet_available": xet_available,
        "xet_version": xet_version,
        "high_performance": high_perf,
        "hint": (
            "Install hf-xet for faster parallel downloads: pip install hf-xet"
            if not xet_available
            else (
                "Set HF_XET_HIGH_PERFORMANCE=1 for max throughput on high-bandwidth links"
                if not high_perf
                else None
            )
        ),
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
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
    # Parallel file downloads inside snapshot_download (standard HF practice)
    os.environ.setdefault("HF_HUB_NUM_THREADS", "8")
    return cache
