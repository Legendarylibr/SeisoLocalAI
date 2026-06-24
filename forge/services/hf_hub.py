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
from functools import partial
from pathlib import Path
from typing import Any, TypeVar

from forge.services.download_progress import ProgressCallback, make_tqdm_class
from seiso.models.catalog import CatalogEntry, get_by_repo
from seiso.models.hub_errors import format_hub_error
from seiso.models.trainable_snapshot import snapshot_has_trainable_weights
from seiso.models.trusted_gguf import (
    base_model_from_tags,
    filter_trusted_gguf_search_results,
    gguf_mirror_candidates,
    is_supported_gguf_repo_candidate,
    is_trusted_gguf_repo,
    rank_trusted_gguf_repos,
)
from seiso.security import sanitize_filename
from seiso.ttl_cache import TtlCache

_HF_API = "https://huggingface.co/api"
_gguf_artifact_cache: TtlCache[str, dict[str, Any]] = TtlCache(ttl_s=86_400.0, max_entries=128)
_gguf_repo_cache: TtlCache[str, str] = TtlCache(ttl_s=86_400.0, max_entries=256)
_repo_gguf_cache: TtlCache[str, bool] = TtlCache(ttl_s=3_600.0, max_entries=512)
_file_size_cache: TtlCache[str, int] = TtlCache(ttl_s=86_400.0, max_entries=512)
_DOWNLOAD_RETRIES = 3
_DOWNLOAD_RETRY_BACKOFF_S = 2.0
_GGUF_SHARD_RE = re.compile(
    r"^(?P<prefix>.+)-(?P<index>\d{5})-of-(?P<total>\d{5})\.gguf$",
    re.I,
)

T = TypeVar("T")


def _snapshot_max_workers() -> int:
    from seiso.models.hf_env import default_hub_num_threads

    raw = os.environ.get("HF_HUB_NUM_THREADS", default_hub_num_threads()).strip()
    try:
        return max(1, min(int(raw), 16))
    except ValueError:
        return 8


def _format_hub_download_error(exc: Exception, *, repo_id: str) -> str:
    return format_hub_error(exc, context="download", repo_id=repo_id)


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
    files = _pick_gguf_files(files, preferred_quant=preferred_quant, repo_id=repo_id)
    return files[0] if files else None


def _pick_gguf_files(
    files: list[str],
    *,
    preferred_quant: str = "Q4_K_M",
    repo_id: str = "",
) -> list[str]:
    ggufs = [
        f
        for f in files
        if f.lower().endswith(".gguf")
        and "mmproj" not in f.lower()
        and not f.lower().startswith("mmproj")
    ]
    if not ggufs:
        return []
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
            pool = active_hits

    shard_groups: dict[tuple[str, str], list[str]] = {}
    for filename in pool:
        name = Path(filename).name
        match = _GGUF_SHARD_RE.match(name)
        if not match:
            continue
        key = (str(Path(filename).parent / match.group("prefix")), match.group("total"))
        shard_groups.setdefault(key, []).append(filename)
    complete_groups: list[list[str]] = []
    for (_prefix, total), group in shard_groups.items():
        try:
            expected = int(total)
        except ValueError:
            continue
        if len(group) == expected:
            complete_groups.append(sorted(group))
    if complete_groups:
        return sorted(complete_groups, key=lambda group: (len(group), len(group[0]), group[0]))[0]

    non_sharded = [filename for filename in pool if not _GGUF_SHARD_RE.match(Path(filename).name)]
    if non_sharded:
        return [sorted(non_sharded, key=len)[0]]
    return []


def _inventory_name(repo_id: str, filename: str) -> Path:
    """Stable symlink path under user models inventory."""
    safe_repo = sanitize_filename(repo_id.replace("/", "--"))
    return Path(safe_repo) / sanitize_filename(Path(filename).name)


def _inventory_name_for_files(repo_id: str, filenames: list[str]) -> Path:
    if len(filenames) == 1:
        return _inventory_name(repo_id, filenames[0])
    first = Path(filenames[0])
    match = _GGUF_SHARD_RE.match(first.name)
    name = match.group("prefix") if match else first.stem
    if first.parent != Path("."):
        name = f"{first.parent.name}-{name}"
    safe_repo = sanitize_filename(repo_id.replace("/", "--"))
    return Path(safe_repo) / sanitize_filename(name)


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
        has_gguf = False
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
                "downloads": row.get("downloads")
                if isinstance(row.get("downloads"), int)
                else None,
                "tags": [t for t in tags if isinstance(t, str)][:4]
                if isinstance(tags, list)
                else [],
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
                "downloads": row.get("downloads")
                if isinstance(row.get("downloads"), int)
                else None,
                "likes": row.get("likes") if isinstance(row.get("likes"), int) else None,
            }
        )
    if trusted_only:
        results = filter_trusted_gguf_search_results(results, base_repo_id=base_repo_id)
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
    if use_cache:
        data = _gguf_artifact_cache.get(cache_key)
        if data is not None:
            return dict(data)

    entry = entry or get_by_repo(catalog_repo_id)
    gguf_repo = resolve_gguf_repo(catalog_repo_id, token=token, revision=revision, entry=entry)
    quant = entry.quant if entry else "Q4_K_M"

    filenames: list[str]
    if not filename:
        files = _list_repo_files(gguf_repo, token=token, revision=revision)
        filenames = _pick_gguf_files(files, preferred_quant=quant, repo_id=catalog_repo_id)
        if not filenames:
            raise ValueError(f"No GGUF files found in {gguf_repo}")
        filename = filenames[0]
    else:
        filenames = [filename]

    size_bytes = sum(
        get_gguf_file_size_bytes(gguf_repo, item, token=token, revision=revision)
        for item in filenames
    )
    info: dict[str, Any] = {
        "catalog_repo": catalog_repo_id,
        "gguf_repo": gguf_repo,
        "filename": filename,
        "filenames": filenames,
        "size_bytes": size_bytes,
        "quant": quant,
    }
    _gguf_artifact_cache.set(cache_key, info)
    return info


def _catalog_base_repo(entry: CatalogEntry, repo_id: str) -> str | None:
    tags = getattr(entry, "tags", ()) or ()
    base = base_model_from_tags(list(tags))
    if base:
        return base
    entry_repo = getattr(entry, "repo_id", None)
    if entry_repo is not None and entry_repo != repo_id:
        return repo_id
    return None


def resolve_gguf_repo(
    repo_id: str,
    *,
    token: str | None = None,
    revision: str = "main",
    entry: CatalogEntry | None = None,
) -> str:
    """Resolve a catalog/base repo to a Hugging Face repo that ships GGUF files."""
    if entry and entry.gguf_repo:
        base_repo = _catalog_base_repo(entry, repo_id)
        if is_trusted_gguf_repo(entry.gguf_repo, base_repo_id=base_repo or repo_id):
            return entry.gguf_repo

    cache_key = f"{repo_id}:{revision}"
    cached = _gguf_repo_cache.get(cache_key)
    if cached is not None:
        return cached

    if repo_has_gguf(repo_id, token=token, revision=revision) and is_trusted_gguf_repo(
        repo_id,
        base_repo_id=repo_id,
    ):
        _gguf_repo_cache.set(cache_key, repo_id)
        return repo_id

    mirror_candidates = rank_trusted_gguf_repos(
        gguf_mirror_candidates(repo_id),
        base_repo_id=repo_id,
    )
    mirror = _first_repo_with_gguf(
        mirror_candidates,
        token=token,
        revision=revision,
    )
    if mirror:
        _gguf_repo_cache.set(cache_key, mirror)
        return mirror

    model_name = repo_id.split("/")[-1]
    needle = model_name.lower().replace("_", "-")
    search_candidates = [
        row["repo_id"]
        for row in search_huggingface_gguf_repos(
            query=model_name,
            limit=12,
            base_repo_id=repo_id,
            trusted_only=True,
        )
        if needle in row["repo_id"].lower().replace("_", "-")
        and is_supported_gguf_repo_candidate(row["repo_id"])
        and is_trusted_gguf_repo(row["repo_id"], base_repo_id=repo_id)
    ]
    resolved = _first_repo_with_gguf(search_candidates, token=token, revision=revision)
    if resolved:
        _gguf_repo_cache.set(cache_key, resolved)
        return resolved

    raise ValueError(
        f"No trusted GGUF quant repo found for {repo_id}. "
        f"Try a reputable mirror such as unsloth/{model_name}-GGUF or bartowski/{model_name}-GGUF."
    )


def download_gguf(
    repo_id: str,
    *,
    cache_dir: Path,
    token: str | None,
    revision: str = "main",
    filename: str | None = None,
    filenames: list[str] | None = None,
    entry: CatalogEntry | None = None,
    inventory_repo_id: str | None = None,
    on_progress: ProgressCallback | None = None,
    total_bytes: int | None = None,
) -> dict[str, Any]:
    from huggingface_hub import hf_hub_download

    quant = entry.quant if entry else "Q4_K_M"
    inv_repo = inventory_repo_id or repo_id

    if filenames:
        filename = filenames[0]
    elif not filename:
        files = _list_repo_files(repo_id, token=token, revision=revision)
        filenames = _pick_gguf_files(files, preferred_quant=quant, repo_id=inv_repo)
        if not filenames:
            raise ValueError(f"No GGUF files found in {repo_id}")
        filename = filenames[0]
    else:
        filenames = [filename]

    if on_progress and (total_bytes is None or total_bytes <= 0):
        try:
            total_bytes = sum(
                get_gguf_file_size_bytes(repo_id, item, token=token, revision=revision)
                for item in filenames
            )
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
    cached_paths: list[Path] = []
    for item in filenames:
        download_kwargs: dict[str, Any] = {
            "repo_id": repo_id,
            "filename": item,
            "revision": revision,
            "token": token,
            "cache_dir": str(cache_dir),
        }
        if on_progress:
            download_kwargs["tqdm_class"] = make_tqdm_class(on_progress)
        cached_paths.append(
            Path(
                _with_download_retries(
                    partial(hf_hub_download, **download_kwargs),  # nosec B615: revision pinned in download_kwargs
                    repo_id=repo_id,
                )
            )
        )

    cached_target = cached_paths[0] if len(cached_paths) == 1 else cached_paths[0].parent

    return {
        "path": str(cached_target),
        "paths": [str(path) for path in cached_paths],
        "filename": filename,
        "filenames": filenames,
        "format": "gguf",
        "repo_id": repo_id,
        "cache_dir": str(cache_dir),
        "inventory_name": str(_inventory_name_for_files(inv_repo, filenames)),
    }


def estimate_snapshot_download_bytes(
    repo_id: str,
    *,
    token: str | None = None,
    revision: str = "main",
) -> int:
    """Estimate bytes needed for a training snapshot (excludes GGUF, markdown, original/)."""
    from huggingface_hub import HfApi

    info = HfApi().model_info(repo_id, revision=revision, token=token)
    total = 0
    for sibling in info.siblings or []:
        name = sibling.rfilename
        if name.endswith(".gguf") or name.endswith(".md") or name.endswith(".h5"):
            continue
        if name.startswith("original/"):
            continue
        total += int(getattr(sibling, "size", None) or 0)
    return total


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
        lambda: snapshot_download(**snapshot_kwargs),  # nosec B615: revision pinned in snapshot_kwargs
        repo_id=repo_id,
    )
    root = Path(path)
    if not snapshot_has_trainable_weights(root):
        from seiso.models.trainable_snapshot import GGUF_ONLY_REPO_MESSAGE

        raise ValueError(f"{GGUF_ONLY_REPO_MESSAGE} Repo: {repo_id}")
    return {
        "path": str(root.resolve()),
        "repo_id": repo_id,
        "format": "safetensors",
        "cache_dir": str(cache_dir),
        "size_bytes": dir_size(root),
    }


def link_inventory(inventory_dir: Path, inventory_name: str, target: Path) -> Path:
    """Symlink cached artifact into user inventory for scanning (no copy)."""
    inventory_dir.mkdir(parents=True, exist_ok=True)
    link = inventory_dir / inventory_name
    link.parent.mkdir(parents=True, exist_ok=True)
    target = target.expanduser().absolute()
    if link.is_symlink() or link.is_file():
        link.unlink()
    elif link.exists() and link.is_dir():
        return link
    link.symlink_to(target, target_is_directory=target.is_dir())
    return link


def dir_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
