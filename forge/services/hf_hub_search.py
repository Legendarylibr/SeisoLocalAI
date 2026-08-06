"""Hugging Face Hub repo discovery and search helpers."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from seiso.models.trusted_gguf import filter_trusted_gguf_search_results
from seiso.ttl_cache import TtlCache

_HF_API = "https://huggingface.co/api"
_gguf_artifact_cache: TtlCache[str, dict[str, Any]] = TtlCache(ttl_s=86_400.0, max_entries=128)
_gguf_repo_cache: TtlCache[str, str] = TtlCache(ttl_s=86_400.0, max_entries=256)
_repo_gguf_cache: TtlCache[str, bool] = TtlCache(ttl_s=3_600.0, max_entries=512)
_file_size_cache: TtlCache[str, int] = TtlCache(ttl_s=86_400.0, max_entries=512)


def _list_repo_files(repo_id: str, *, token: str | None, revision: str) -> list[str]:
    from huggingface_hub import list_repo_files

    return list_repo_files(repo_id, token=token, revision=revision)


def repo_has_gguf(repo_id: str, *, token: str | None = None, revision: str = "main") -> bool:
    cache_key = f"{repo_id}:{revision}"
    cached = _repo_gguf_cache.get(cache_key)
    if cached is not None:
        return cached
    try:
        files = _list_repo_files(repo_id, token=token, revision=revision)
        has_gguf = any(f.lower().endswith(".gguf") for f in files)
    except Exception:
        return False
    _repo_gguf_cache.set(cache_key, has_gguf)
    return has_gguf


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
    if not url.startswith("https://"):
        return []
    request = urllib.request.Request(url, headers={"User-Agent": "seiso-forge/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=15.0) as response:  # nosec B310: only https://huggingface.co URLs
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
                "downloads": (
                    row.get("downloads") if isinstance(row.get("downloads"), int) else None
                ),
                "tags": (
                    [t for t in tags if isinstance(t, str)][:4] if isinstance(tags, list) else []
                ),
            }
        )
    return results


def search_huggingface_gguf_repos(
    *,
    query: str,
    limit: int = 8,
    base_repo_id: str | None = None,
    trusted_only: bool = True,
) -> list[dict[str, Any]]:
    """Search Hugging Face Hub for GGUF repos (no token required)."""
    q = query.strip()
    if not q:
        return []
    fetch_limit = max(1, min(limit * 4 if trusted_only else limit, 50))
    params = urllib.parse.urlencode(
        {
            "search": q,
            "filter": "gguf",
            "limit": fetch_limit,
            "full": "false",
        }
    )
    url = f"{_HF_API}/models?{params}"
    if not url.startswith("https://"):
        return []
    request = urllib.request.Request(url, headers={"User-Agent": "seiso-forge/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=15.0) as response:  # nosec B310: only https://huggingface.co URLs
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
        if not isinstance(model_id, str):
            continue
        results.append(
            {
                "repo_id": model_id,
                "downloads": (
                    row.get("downloads") if isinstance(row.get("downloads"), int) else None
                ),
                "likes": (row.get("likes") if isinstance(row.get("likes"), int) else None),
            }
        )
    results = filter_trusted_gguf_search_results(results, base_repo_id=base_repo_id)
    if not trusted_only:
        results.sort(
            key=lambda row: (
                -int(row.get("downloads") or 0),
                str(row.get("repo_id") or "").lower(),
            )
        )
    return results[: max(1, min(limit, 25))]


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
    results: dict[str, bool] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(repo_has_gguf, repo_id, token=token, revision=revision): repo_id
            for repo_id in repo_ids
        }
        for future in as_completed(futures):
            repo_id = futures[future]
            try:
                results[repo_id] = bool(future.result())
            except Exception:
                results[repo_id] = False
    for repo_id in repo_ids:
        if results.get(repo_id):
            return repo_id
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
    cached = _file_size_cache.get(cache_key)
    if cached is not None:
        return cached

    from huggingface_hub import get_hf_file_metadata, hf_hub_url

    url = hf_hub_url(repo_id, filename, repo_type="model", revision=revision)
    meta = get_hf_file_metadata(url, token=token)
    size_bytes = int(meta.size or 0)
    _file_size_cache.set(cache_key, size_bytes)
    return size_bytes
