"""Persist successful llama.cpp load profiles so the next warm avoids the retry ladder."""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_CACHE_LOCK = threading.Lock()
_CACHE_NAME = "llama_load_profiles.json"
_MAX_ENTRIES = 64
# Profile fields that affect correctness / VRAM fit.
_PROFILE_KEYS = (
    "n_gpu_layers",
    "n_ctx",
    "n_batch",
    "n_ubatch",
    "flash_attn",
    "offload_kqv",
    "op_offload",
    "type_k",
    "type_v",
    "swa_full",
    "load_tier",
)


def _cache_path() -> Path:
    from seiso.security import resolve_data_dir

    return resolve_data_dir() / "cache" / _CACHE_NAME


def _path_fingerprint(model_path: str) -> str:
    path = Path(model_path).expanduser()
    try:
        resolved = str(path.resolve())
        st = path.stat()
        return f"{resolved}:{st.st_mtime_ns}:{st.st_size}"
    except OSError:
        return str(path)


def profile_cache_key(model_path: str, *, n_ctx: int, load_tier: str = "normal") -> str:
    """Stable key for a model file + context bucket + load tier."""
    ctx_bucket = max(2048, int(n_ctx))
    # Coarse bucket so multi-turn growth within a band reuses one profile.
    ctx_bucket = (ctx_bucket + 2047) // 2048 * 2048
    return f"{_path_fingerprint(model_path)}|ctx={ctx_bucket}|tier={load_tier}"


def _read_store(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"version": 1, "profiles": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"version": 1, "profiles": {}}
    if not isinstance(data, dict):
        return {"version": 1, "profiles": {}}
    profiles = data.get("profiles")
    if not isinstance(profiles, dict):
        data["profiles"] = {}
    return data


def _write_store(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def get_cached_load_profile(
    model_path: str,
    *,
    n_ctx: int,
    load_tier: str = "normal",
) -> dict[str, Any] | None:
    """Return a previously successful load profile, or None."""
    key = profile_cache_key(model_path, n_ctx=n_ctx, load_tier=load_tier)
    path = _cache_path()
    with _CACHE_LOCK:
        store = _read_store(path)
        entry = store.get("profiles", {}).get(key)
    if not isinstance(entry, dict):
        return None
    profile = entry.get("profile")
    if not isinstance(profile, dict):
        return None
    # Only return known keys so stale extras cannot break Llama().
    out = {k: profile[k] for k in _PROFILE_KEYS if k in profile}
    if "n_gpu_layers" not in out:
        return None
    return out


def save_cached_load_profile(
    model_path: str,
    *,
    n_ctx: int,
    load_tier: str = "normal",
    profile: dict[str, Any],
) -> None:
    """Remember a successful load so the next process can skip failed attempts."""
    key = profile_cache_key(model_path, n_ctx=n_ctx, load_tier=load_tier)
    slim: dict[str, Any] = {}
    for k in _PROFILE_KEYS:
        if k not in profile or profile[k] is None:
            continue
        value = profile[k]
        # JSON-safe only (GGML type enums are ints).
        if isinstance(value, (bool, int, float, str)):
            slim[k] = value
        else:
            try:
                slim[k] = int(value)
            except (TypeError, ValueError):
                continue
    if "n_gpu_layers" not in slim:
        return
    path = _cache_path()
    with _CACHE_LOCK:
        store = _read_store(path)
        profiles: dict[str, Any] = store.setdefault("profiles", {})
        profiles[key] = {
            "profile": slim,
            "saved_at": time.time(),
            "model": Path(model_path).name,
        }
        # Evict oldest entries when over capacity.
        if len(profiles) > _MAX_ENTRIES:
            ordered = sorted(
                profiles.items(),
                key=lambda item: float(item[1].get("saved_at") or 0.0),
            )
            for stale_key, _ in ordered[: len(profiles) - _MAX_ENTRIES]:
                profiles.pop(stale_key, None)
        try:
            _write_store(path, store)
        except (OSError, TypeError, ValueError) as exc:
            logger.debug("Could not persist llama load profile: %s", exc)


def clear_load_profile_cache() -> None:
    """Drop all cached profiles (tests / after env changes)."""
    path = _cache_path()
    with _CACHE_LOCK:
        try:
            if path.is_file():
                path.unlink()
        except OSError:
            pass


def profile_from_load_kwargs(
    load_kwargs: dict[str, Any],
    *,
    layers: int,
    load_tier: str,
) -> dict[str, Any]:
    """Build a cacheable profile dict from the kwargs that succeeded."""
    out: dict[str, Any] = {
        "n_gpu_layers": int(layers),
        "load_tier": load_tier,
    }
    for key in (
        "n_ctx",
        "n_batch",
        "n_ubatch",
        "flash_attn",
        "offload_kqv",
        "op_offload",
        "type_k",
        "type_v",
        "swa_full",
    ):
        if key in load_kwargs and load_kwargs[key] is not None:
            out[key] = load_kwargs[key]
    return out
