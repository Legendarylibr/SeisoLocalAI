"""Shared Hugging Face Hub cache configuration for downloads and training."""

from __future__ import annotations

import contextlib
import logging
import os
from pathlib import Path
from typing import Any

from seiso.env import env_bool
from seiso.security import resolve_data_dir

logger = logging.getLogger(__name__)


def resolve_hf_cache_dir(data_dir: Path | None = None) -> Path:
    """Directory used by huggingface_hub and transformers."""
    root = resolve_data_dir(data_dir)
    if raw := os.environ.get("HUGGINGFACE_HUB_CACHE"):
        candidate = Path(raw).expanduser().resolve()
        try:
            candidate.relative_to(root.resolve())
            return candidate
        except ValueError:
            pass
    if raw := os.environ.get("HF_HOME"):
        candidate = Path(raw).expanduser().resolve()
        try:
            candidate.relative_to(root.resolve())
            return candidate / "hub"
        except ValueError:
            pass
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

    high_perf = env_bool("HF_XET_HIGH_PERFORMANCE", False)
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


def _read_hub_token() -> str | None:
    """Read a Hugging Face token from env or the default CLI login locations."""
    for key in ("HUGGING_FACE_HUB_TOKEN", "HF_TOKEN"):
        raw = os.environ.get(key, "").strip()
        if raw:
            return raw
    home = Path.home()
    for path in (
        home / ".cache" / "huggingface" / "token",
        home / ".huggingface" / "token",
    ):
        if path.is_file():
            raw = path.read_text(encoding="utf-8").strip()
            if raw:
                return raw
    return None


def configure_hf_hub_auth(token: str | None = None) -> str | None:
    """
    Ensure gated Hub models work when Seiso relocates HF_HOME away from ~/.cache.

    huggingface_hub only reads ``$HF_HOME/token`` when HF_HOME is set; it does not
    fall back to ``~/.cache/huggingface/token``. Mirror the CLI token into env (and
    the active HF_HOME when configured) before model/dataset downloads.
    """
    resolved = (token or _read_hub_token() or "").strip() or None
    if not resolved:
        return None
    try:
        from huggingface_hub import HfApi

        HfApi(token=resolved).whoami()
    except Exception:
        logger.warning(
            "Hugging Face token is missing or invalid — gated models and some datasets "
            "will fail until you run `hf auth login` or save a token in Seiso Settings."
        )
        return None
    os.environ["HF_TOKEN"] = resolved
    os.environ["HUGGING_FACE_HUB_TOKEN"] = resolved
    hf_home = os.environ.get("HF_HOME", "").strip()
    if hf_home:
        token_path = Path(hf_home).expanduser() / "token"
        try:
            token_path.parent.mkdir(parents=True, exist_ok=True)
            if (
                not token_path.exists()
                or token_path.read_text(encoding="utf-8").strip() != resolved
            ):
                token_path.write_text(resolved + "\n", encoding="utf-8")
                token_path.chmod(0o600)
        except OSError:
            pass
    return resolved


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
    hf_home = root / "hf_home"
    cache = root / "hf_cache"
    if raw_cache := os.environ.get("HUGGINGFACE_HUB_CACHE"):
        candidate = Path(raw_cache).expanduser().resolve()
        try:
            candidate.relative_to(root.resolve())
            cache = candidate
        except ValueError:
            # External caches break later assert_user_path on inventory symlinks.
            logger.warning(
                "Ignoring HUGGINGFACE_HUB_CACHE outside SEISO_DATA_DIR (%s); "
                "using %s so downloads stay sandboxed",
                candidate,
                cache,
            )
    cache.mkdir(parents=True, exist_ok=True)
    # Always relocate Hub state under Seiso's data dir (do not leave a stale HF_HOME).
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache)
    xet_cache = Path(os.environ.get("HF_XET_CACHE", root / "hf_xet_cache")).expanduser()
    xet_cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_XET_CACHE"] = str(xet_cache)
    # huggingface_hub may have cached default paths at import time — refresh when loaded.
    with contextlib.suppress(Exception):
        import huggingface_hub.constants as hf_constants

        hf_constants.HF_HOME = os.environ["HF_HOME"]
        hf_constants.HF_HUB_CACHE = os.environ["HUGGINGFACE_HUB_CACHE"]
        hf_constants.default_cache_path = cache
    # hf-xet writes logs under HF_HOME/xet/logs; create it early so failures are actionable.
    (hf_home / "xet" / "logs").mkdir(parents=True, exist_ok=True)
    # Generous defaults for large GGUF / safetensors downloads
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
    # More parallel shard downloads when hf-xet is available.
    os.environ.setdefault("HF_HUB_NUM_THREADS", default_hub_num_threads())
    # hf_transfer was removed; the old env var is a silent no-op and can hide missing XET tuning.
    os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
    if _xet_available():
        os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
    configure_hf_hub_auth()
    return cache
