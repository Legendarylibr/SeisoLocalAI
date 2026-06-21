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


def default_hub_num_threads() -> str:
    return "16" if _xet_available() else "8"


def default_hub_download_timeout() -> str:
    return "600"


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
    num_threads = os.environ.get("HF_HUB_NUM_THREADS", default_hub_num_threads()).strip()
    download_timeout = os.environ.get(
        "HF_HUB_DOWNLOAD_TIMEOUT", default_hub_download_timeout()
    ).strip()
    hf_home = os.environ.get("HF_HOME", "").strip()
    xet_cache = os.environ.get("HF_XET_CACHE", "").strip()
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
        "hf_home": hf_home or None,
        "xet_cache": xet_cache or None,
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
    - Xet cache/log state under the Seiso data dir instead of the user's global cache
    """
    root = resolve_data_dir(data_dir)
    hf_home = Path(os.environ.get("HF_HOME", root / "hf_home")).expanduser()
    os.environ.setdefault("HF_HOME", str(hf_home))
    if raw_cache := os.environ.get("HUGGINGFACE_HUB_CACHE"):
        cache = Path(raw_cache).expanduser()
    else:
        cache = root / "hf_cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache))
    xet_cache = Path(os.environ.get("HF_XET_CACHE", root / "hf_xet_cache")).expanduser()
    xet_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_XET_CACHE", str(xet_cache))
    # hf-xet writes logs under HF_HOME/xet/logs; create it early so failures are actionable.
    (hf_home / "xet" / "logs").mkdir(parents=True, exist_ok=True)
    # Generous defaults for large GGUF / safetensors downloads
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
    # More parallel shard downloads when hf-xet is available.
    os.environ.setdefault("HF_HUB_NUM_THREADS", default_hub_num_threads())
    if _xet_available():
        os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    return cache
