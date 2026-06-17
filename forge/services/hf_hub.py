"""Hugging Face Hub downloads via shared cache — no duplicate blobs on disk."""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, TypeVar

from forge.services.download_progress import ProgressCallback, make_tqdm_class
from seiso.models.catalog import CatalogEntry, get_by_repo
from seiso.security import sanitize_filename

_HF_API = "https://huggingface.co/api"
_GGUF_ARTIFACT_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_GGUF_ARTIFACT_TTL_S = 86_400.0  # 24h — mirror filenames rarely change
_GGUF_REPO_CACHE: dict[str, tuple[float, str]] = {}
_GGUF_REPO_TTL_S = 86_400.0
_REPO_GGUF_CACHE: dict[str, tuple[float, bool]] = {}
_REPO_GGUF_TTL_S = 3_600.0
_FILE_SIZE_CACHE: dict[str, tuple[float, int]] = {}
_FILE_SIZE_TTL_S = 86_400.0
_DOWNLOAD_RETRIES = 3
_DOWNLOAD_RETRY_BACKOFF_S = 2.0

T = TypeVar("T")


def _snapshot_max_workers() -> int:
    raw = os.environ.get("HF_HUB_NUM_THREADS", "8").strip()
    try:
        return max(1, min(int(raw), 16))
    except ValueError:
        return 8


def _format_hub_download_error(exc: Exception, *, repo_id: str) -> str:
    msg = str(exc).strip() or exc.__class__.__name__
    lowered = msg.lower()
    if "401" in msg or "403" in msg or "gated" in lowered or "authorized" in lowered:
        return (
            f"Access denied for {repo_id}. This model may be gated — "
            "save a Hugging Face token in Settings or run `hf auth login`."
        )
    if "404" in msg or "not found" in lowered:
        return f"Model repo not found on Hugging Face Hub: {repo_id}"
    if "connection" in lowered or "network" in lowered or "resolve" in lowered:
        return f"Cannot reach huggingface.co while downloading {repo_id}. Check your network."
    if "timeout" in lowered or "timed out" in lowered:
        return f"Download timed out for {repo_id}. Retry or set HF_HUB_DOWNLOAD_TIMEOUT higher."
    return f"Hub download failed for {repo_id}: {msg}"


def _with_download_retries(fn: Callable[[], T], *, repo_id: str) -> T:
    """Retry transient Hub download failures with exponential backoff."""
    last_exc: Exception | None = None
    for attempt in range(_DOWNLOAD_RETRIES):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            msg = str(exc).lower()
            if any(code in str(exc) for code in ("401", "403", "404")):
                break
            if attempt + 1 >= _DOWNLOAD_RETRIES:
                break
            if not any(
                hint in msg
                for hint in (
                    "timeout",
                    "timed out",
                    "connection",
                    "network",
                    "temporar",
                    "503",
                    "502",
                    "429",
                )
            ):
                break
            time.sleep(_DOWNLOAD_RETRY_BACKOFF_S * (2**attempt))
    assert last_exc is not None
    raise ValueError(_format_hub_download_error(last_exc, repo_id=repo_id)) from last_exc


def _pick_gguf_file(
    files: list[str],
    *,
    preferred_quant: str = "Q4_K_M",
    repo_id: str = "",
) -> str | None:
    ggufs = [
        f
        for f in files
        if f.lower().endswith(".gguf")
        and "mmproj" not in f.lower()
        and not f.lower().startswith("mmproj")
    ]
    if not ggufs:
        return None
    preferred_quant = preferred_quant.upper()

    def quant_matches(candidates: list[str]) -> list[str]:
        exact = [f for f in candidates if preferred_quant in f.upper()]
        if exact:
            return exact
        for hint in ("Q4_K_M", "Q5_K_M", "Q4_0", "Q8_0", "IQ4_XS"):
            matched = [f for f in candidates if hint in f.upper()]
            if matched:
                return matched
        return candidates

    pool = quant_matches(ggufs)

    moe_match = re.search(r"a(\d+(?:\.\d+)?)b", repo_id, re.I)
    if moe_match:
        active = moe_match.group(0).lower()
        active_hits = [f for f in pool if active in f.lower().replace("_", "-")]
        if active_hits:
            return sorted(active_hits, key=len)[0]

    return sorted(pool, key=len)[0]


def _inventory_name(repo_id: str, filename: str) -> Path:
    """Stable symlink path under user models inventory."""
    safe_repo = sanitize_filename(repo_id.replace("/", "--"))
    return Path(safe_repo) / sanitize_filename(Path(filename).name)


def _list_repo_files(repo_id: str, *, token: str | None, revision: str) -> list[str]:
    from huggingface_hub import list_repo_files

    return list_repo_files(repo_id, token=token, revision=revision)


def repo_has_gguf(repo_id: str, *, token: str | None = None, revision: str = "main") -> bool:
    cache_key = f"{repo_id}:{revision}"
    now = time.time()
    cached = _REPO_GGUF_CACHE.get(cache_key)
    if cached and now - cached[0] < _REPO_GGUF_TTL_S:
        return cached[1]
    try:
        files = _list_repo_files(repo_id, token=token, revision=revision)
        has_gguf = any(f.lower().endswith(".gguf") for f in files)
    except Exception:
        has_gguf = False
    _REPO_GGUF_CACHE[cache_key] = (now, has_gguf)
    return has_gguf


def _gguf_mirror_candidates(repo_id: str) -> list[str]:
    """Common community GGUF mirrors for a base-model repo id."""
    model_name = repo_id.split("/")[-1]
    title = re.sub(r"(^|[-_/])([a-z])", lambda m: m.group(1) + m.group(2).upper(), model_name)
    mirrors = [
        f"bartowski/{model_name}-GGUF",
        f"bartowski/{title}-GGUF",
        f"unsloth/{model_name}-GGUF",
        f"QuantFactory/{model_name}-GGUF",
        f"QuantFactory/{title}-GGUF",
        f"lmstudio-community/{model_name}-GGUF",
        f"lmstudio-community/{title}-GGUF",
    ]
    if "Qwen" in repo_id:
        mirrors.insert(2, f"bartowski/Qwen_{model_name}-GGUF")
    seen: set[str] = set()
    ordered: list[str] = []
    for candidate in mirrors:
        if candidate not in seen:
            seen.add(candidate)
            ordered.append(candidate)
    return ordered


def search_huggingface_datasets(*, query: str, limit: int = 12) -> list[dict[str, Any]]:
    """Search Hugging Face Hub for datasets (no token required)."""
    q = query.strip()
    if not q:
        return []
    params = urllib.parse.urlencode(
        {
            "search": q,
            "limit": max(1, min(limit, 25)),
            "full": "false",
        }
    )
    url = f"{_HF_API}/datasets?{params}"
    request = urllib.request.Request(url, headers={"User-Agent": "seiso-forge/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=15.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return []
    if not isinstance(payload, list):
        return []
    results: list[dict[str, Any]] = []
    for row in payload:
        if not isinstance(row, dict):
            continue
        dataset_id = row.get("id")
        if not isinstance(dataset_id, str):
            continue
        tags = row.get("tags")
        results.append(
            {
                "repo_id": dataset_id,
                "name": dataset_id.split("/")[-1],
                "downloads": row.get("downloads") if isinstance(row.get("downloads"), int) else None,
                "tags": [t for t in tags if isinstance(t, str)][:4] if isinstance(tags, list) else [],
            }
        )
    return results


def search_huggingface_gguf_repos(*, query: str, limit: int = 8) -> list[dict[str, Any]]:
    """Search Hugging Face Hub for GGUF repos (no token required)."""
    q = query.strip()
    if not q:
        return []
    params = urllib.parse.urlencode(
        {
            "search": q,
            "filter": "gguf",
            "limit": max(1, min(limit, 25)),
            "full": "false",
        }
    )
    url = f"{_HF_API}/models?{params}"
    request = urllib.request.Request(url, headers={"User-Agent": "seiso-forge/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=15.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return []
    if not isinstance(payload, list):
        return []
    results: list[dict[str, Any]] = []
    for row in payload:
        if not isinstance(row, dict):
            continue
        model_id = row.get("id") or row.get("modelId")
        if isinstance(model_id, str):
            results.append({"repo_id": model_id})
    return results


def _first_repo_with_gguf(
    repo_ids: list[str],
    *,
    token: str | None = None,
    revision: str = "main",
    max_workers: int = 4,
) -> str | None:
    """Return the first repo id that contains GGUF files (checked in parallel)."""
    if not repo_ids:
        return None
    if len(repo_ids) == 1:
        return repo_ids[0] if repo_has_gguf(repo_ids[0], token=token, revision=revision) else None

    workers = min(max_workers, len(repo_ids))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(repo_has_gguf, repo_id, token=token, revision=revision): repo_id
            for repo_id in repo_ids
        }
        for future in as_completed(futures):
            repo_id = futures[future]
            try:
                if future.result():
                    for pending in futures:
                        pending.cancel()
                    return repo_id
            except Exception:
                continue
    return None


def get_gguf_file_size_bytes(
    repo_id: str,
    filename: str,
    *,
    token: str | None = None,
    revision: str = "main",
) -> int:
    """Return on-disk byte size for a single Hub file (uses HF metadata API)."""
    cache_key = f"{repo_id}:{filename}:{revision}"
    now = time.time()
    cached = _FILE_SIZE_CACHE.get(cache_key)
    if cached and now - cached[0] < _FILE_SIZE_TTL_S:
        return cached[1]

    from huggingface_hub import get_hf_file_metadata, hf_hub_url

    url = hf_hub_url(repo_id, filename, repo_type="model", revision=revision)
    meta = get_hf_file_metadata(url, token=token)
    size_bytes = int(meta.size or 0)
    _FILE_SIZE_CACHE[cache_key] = (now, size_bytes)
    return size_bytes


def resolve_gguf_artifact(
    catalog_repo_id: str,
    *,
    token: str | None = None,
    revision: str = "main",
    entry: CatalogEntry | None = None,
    filename: str | None = None,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Resolve GGUF mirror, preferred quant file, and download size for a catalog repo."""
    cache_key = f"{catalog_repo_id}:{filename or ''}:{revision}"
    now = time.time()
    if use_cache and cache_key in _GGUF_ARTIFACT_CACHE:
        ts, data = _GGUF_ARTIFACT_CACHE[cache_key]
        if now - ts < _GGUF_ARTIFACT_TTL_S:
            return dict(data)

    entry = entry or get_by_repo(catalog_repo_id)
    gguf_repo = resolve_gguf_repo(catalog_repo_id, token=token, revision=revision, entry=entry)
    quant = entry.quant if entry else "Q4_K_M"

    if not filename:
        files = _list_repo_files(gguf_repo, token=token, revision=revision)
        filename = _pick_gguf_file(files, preferred_quant=quant, repo_id=catalog_repo_id)
        if not filename:
            raise ValueError(f"No GGUF files found in {gguf_repo}")

    size_bytes = get_gguf_file_size_bytes(gguf_repo, filename, token=token, revision=revision)
    info: dict[str, Any] = {
        "catalog_repo": catalog_repo_id,
        "gguf_repo": gguf_repo,
        "filename": filename,
        "size_bytes": size_bytes,
        "quant": quant,
    }
    _GGUF_ARTIFACT_CACHE[cache_key] = (now, info)
    return info


def resolve_gguf_repo(
    repo_id: str,
    *,
    token: str | None = None,
    revision: str = "main",
    entry: CatalogEntry | None = None,
) -> str:
    """Resolve a catalog/base repo to a Hugging Face repo that ships GGUF files."""
    entry = entry or get_by_repo(repo_id)
    if entry and entry.gguf_repo:
        return entry.gguf_repo

    cache_key = f"{repo_id}:{revision}"
    now = time.time()
    cached = _GGUF_REPO_CACHE.get(cache_key)
    if cached and now - cached[0] < _GGUF_REPO_TTL_S:
        return cached[1]

    if repo_has_gguf(repo_id, token=token, revision=revision):
        _GGUF_REPO_CACHE[cache_key] = (now, repo_id)
        return repo_id

    mirror = _first_repo_with_gguf(
        _gguf_mirror_candidates(repo_id),
        token=token,
        revision=revision,
    )
    if mirror:
        _GGUF_REPO_CACHE[cache_key] = (now, mirror)
        return mirror

    model_name = repo_id.split("/")[-1]
    needle = model_name.lower().replace("_", "-")
    search_candidates = [
        row["repo_id"]
        for row in search_huggingface_gguf_repos(query=model_name, limit=10)
        if needle in row["repo_id"].lower().replace("_", "-")
    ]
    resolved = _first_repo_with_gguf(search_candidates, token=token, revision=revision)
    if resolved:
        _GGUF_REPO_CACHE[cache_key] = (now, resolved)
        return resolved

    raise ValueError(
        f"No GGUF quant repo found for {repo_id}. "
        f"Try a GGUF mirror such as bartowski/{model_name}-GGUF."
    )


def download_gguf(
    repo_id: str,
    *,
    cache_dir: Path,
    token: str | None,
    revision: str = "main",
    filename: str | None = None,
    entry: CatalogEntry | None = None,
    inventory_repo_id: str | None = None,
    on_progress: ProgressCallback | None = None,
    total_bytes: int | None = None,
) -> dict[str, Any]:
    from huggingface_hub import hf_hub_download

    entry = entry or get_by_repo(repo_id)
    quant = entry.quant if entry else "Q4_K_M"
    inv_repo = inventory_repo_id or repo_id

    if not filename:
        files = _list_repo_files(repo_id, token=token, revision=revision)
        filename = _pick_gguf_file(files, preferred_quant=quant, repo_id=inv_repo)
        if not filename:
            raise ValueError(f"No GGUF files found in {repo_id}")

    if on_progress and (total_bytes is None or total_bytes <= 0):
        try:
            total_bytes = get_gguf_file_size_bytes(repo_id, filename, token=token, revision=revision)
        except Exception:
            total_bytes = 0
        if total_bytes > 0:
            on_progress(
                {
                    "phase": "download",
                    "label": f"Downloading {filename}",
                    "repo_id": repo_id,
                    "total_bytes": total_bytes,
                    "bytes": 0,
                    "percent": 0,
                    "eta_seconds": None,
                    "speed_bps": 0,
                }
            )

    cache_dir.mkdir(parents=True, exist_ok=True)
    download_kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "filename": filename,
        "revision": revision,
        "token": token,
        "cache_dir": str(cache_dir),
    }
    if on_progress:
        download_kwargs["tqdm_class"] = make_tqdm_class(on_progress)
    cached_path = Path(
        _with_download_retries(
            lambda: hf_hub_download(**download_kwargs),
            repo_id=repo_id,
        )
    )

    return {
        "path": str(cached_path.resolve()),
        "filename": filename,
        "format": "gguf",
        "repo_id": repo_id,
        "cache_dir": str(cache_dir),
        "inventory_name": str(_inventory_name(inv_repo, filename)),
    }


def download_training_snapshot(
    repo_id: str,
    *,
    cache_dir: Path,
    token: str | None,
    revision: str = "main",
    on_progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    from huggingface_hub import snapshot_download

    cache_dir.mkdir(parents=True, exist_ok=True)
    snapshot_kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "revision": revision,
        "token": token,
        "cache_dir": str(cache_dir),
        "ignore_patterns": ["*.md", "*.h5", "original/*", "*.gguf"],
        "max_workers": _snapshot_max_workers(),
    }
    if on_progress:
        snapshot_kwargs["tqdm_class"] = make_tqdm_class(on_progress)
    path = _with_download_retries(
        lambda: snapshot_download(**snapshot_kwargs),
        repo_id=repo_id,
    )
    root = Path(path)
    return {
        "path": str(root.resolve()),
        "repo_id": repo_id,
        "format": "safetensors",
        "cache_dir": str(cache_dir),
        "size_bytes": _dir_size(root),
    }


def link_inventory(inventory_dir: Path, inventory_name: str, target: Path) -> Path:
    """Symlink cached artifact into user inventory for scanning (no copy)."""
    inventory_dir.mkdir(parents=True, exist_ok=True)
    link = inventory_dir / inventory_name
    link.parent.mkdir(parents=True, exist_ok=True)
    target = target.resolve()
    if link.is_symlink() or link.is_file():
        link.unlink()
    elif link.exists() and link.is_dir():
        return link
    link.symlink_to(target, target_is_directory=target.is_dir())
    return link


def _dir_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
