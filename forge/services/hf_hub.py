"""Hugging Face Hub downloads via shared cache — no duplicate blobs on disk."""

from __future__ import annotations

import os
import shutil
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any, TypeVar

from forge.services.download_progress import ProgressCallback, make_tqdm_class
from forge.services.hf_hub_gguf_select import (
    _complete_shard_group_for,
    _gguf_artifact_likely_vision,
    _inventory_name_for_files,
    _pick_gguf_file,
    _pick_gguf_files,
    _pick_mmproj_file,
    list_complete_gguf_file_groups,
)
from forge.services.hf_hub_search import (
    _first_repo_with_gguf,
    _gguf_artifact_cache,
    _gguf_repo_cache,
    _list_repo_files,
    get_gguf_file_size_bytes,
    repo_has_gguf,
    search_huggingface_datasets,
    search_huggingface_gguf_repos,
)
from seiso.io.files import model_weight_size_bytes
from seiso.models.catalog import CatalogEntry, get_by_repo
from seiso.models.hub_errors import format_hub_error
from seiso.models.trainable_snapshot import snapshot_has_trainable_weights
from seiso.models.trusted_gguf import gguf_mirror_candidates

_DOWNLOAD_RETRIES = 3
_DOWNLOAD_RETRY_BACKOFF_S = 2.0

T = TypeVar("T")

__all__ = [
    "_complete_shard_group_for",
    "_format_hub_download_error",
    "_pick_gguf_file",
    "_pick_gguf_files",
    "download_gguf",
    "download_training_snapshot",
    "estimate_snapshot_download_bytes",
    "get_gguf_file_size_bytes",
    "link_inventory",
    "list_complete_gguf_file_groups",
    "repo_has_gguf",
    "resolve_gguf_artifact",
    "resolve_gguf_repo",
    "search_huggingface_datasets",
    "search_huggingface_gguf_repos",
]


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
    raise ValueError(
        _format_hub_download_error(last_exc, repo_id=repo_id)
    ) from last_exc


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
    gguf_repo = resolve_gguf_repo(
        catalog_repo_id, token=token, revision=revision, entry=entry
    )
    quant = entry.quant if entry else "Q4_K_M"

    files = _list_repo_files(gguf_repo, token=token, revision=revision)
    filenames: list[str]
    if not filename:
        filenames = _pick_gguf_files(
            files, preferred_quant=quant, repo_id=catalog_repo_id
        )
        if not filenames:
            raise ValueError(f"No GGUF files found in {gguf_repo}")
        filename = filenames[0]
    else:
        filenames = _complete_shard_group_for(files, filename)
        filename = filenames[0]

    size_bytes = sum(
        get_gguf_file_size_bytes(gguf_repo, item, token=token, revision=revision)
        for item in filenames
    )
    mmproj_filename: str | None = None
    if _gguf_artifact_likely_vision(
        catalog_repo_id,
        gguf_filename=filename,
        entry=entry,
    ):
        mmproj_filename = _pick_mmproj_file(files, preferred_quant=quant)
    if mmproj_filename:
        size_bytes += get_gguf_file_size_bytes(
            gguf_repo, mmproj_filename, token=token, revision=revision
        )
    info: dict[str, Any] = {
        "catalog_repo": catalog_repo_id,
        "gguf_repo": gguf_repo,
        "filename": filename,
        "filenames": filenames,
        "size_bytes": size_bytes,
        "quant": quant,
    }
    if mmproj_filename:
        info["mmproj_filename"] = mmproj_filename
    _gguf_artifact_cache.set(cache_key, info)
    return info


def resolve_gguf_repo(
    repo_id: str,
    *,
    token: str | None = None,
    revision: str = "main",
    entry: CatalogEntry | None = None,
) -> str:
    """Resolve a catalog/base repo to a Hugging Face repo that ships GGUF files."""
    if (
        entry
        and entry.gguf_repo
        and repo_has_gguf(
            entry.gguf_repo,
            token=token,
            revision=revision,
        )
    ):
        return entry.gguf_repo

    cache_key = f"{repo_id}:{revision}"
    cached = _gguf_repo_cache.get(cache_key)
    if cached is not None:
        return cached

    if repo_has_gguf(repo_id, token=token, revision=revision):
        _gguf_repo_cache.set(cache_key, repo_id)
        return repo_id

    mirror_candidates = gguf_mirror_candidates(repo_id)
    mirror = _first_repo_with_gguf(
        mirror_candidates,
        token=token,
        revision=revision,
    )
    if mirror:
        _gguf_repo_cache.set(cache_key, mirror)
        return mirror

    model_name = repo_id.split("/")[-1]
    search_candidates = [
        row["repo_id"]
        for row in search_huggingface_gguf_repos(
            query=model_name,
            limit=12,
            base_repo_id=repo_id,
            trusted_only=True,
        )
    ]
    resolved = _first_repo_with_gguf(search_candidates, token=token, revision=revision)
    if resolved:
        _gguf_repo_cache.set(cache_key, resolved)
        return resolved

    raise ValueError(f"No GGUF files found for {repo_id} on Hugging Face Hub.")


def download_gguf(
    repo_id: str,
    *,
    cache_dir: Path,
    token: str | None,
    revision: str = "main",
    filename: str | None = None,
    filenames: list[str] | None = None,
    mmproj_filename: str | None = None,
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

    download_names = list(filenames)
    if mmproj_filename and mmproj_filename not in download_names:
        download_names.append(mmproj_filename)

    if on_progress and (total_bytes is None or total_bytes <= 0):
        try:
            total_bytes = sum(
                get_gguf_file_size_bytes(repo_id, item, token=token, revision=revision)
                for item in download_names
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
    cached_by_name: dict[str, Path] = {}
    for item in download_names:
        download_kwargs: dict[str, Any] = {
            "repo_id": repo_id,
            "filename": item,
            "revision": revision,
            "token": token,
            "cache_dir": str(cache_dir),
        }
        if on_progress:
            download_kwargs["tqdm_class"] = make_tqdm_class(on_progress)
        cached = Path(
            _with_download_retries(
                partial(
                    hf_hub_download, **download_kwargs
                ),  # nosec B615: revision pinned in download_kwargs
                repo_id=repo_id,
            )
        )
        cached_paths.append(cached)
        cached_by_name[item] = cached

    cached_target = (
        cached_paths[0] if len(cached_paths) == 1 else cached_paths[0].parent
    )

    result: dict[str, Any] = {
        "path": str(cached_target),
        "paths": [str(path) for path in cached_paths],
        "filename": filename,
        "filenames": filenames,
        "format": "gguf",
        "repo_id": repo_id,
        "cache_dir": str(cache_dir),
        "inventory_name": str(_inventory_name_for_files(inv_repo, filenames)),
    }
    if mmproj_filename:
        mmproj_cached = cached_by_name.get(mmproj_filename)
        if mmproj_cached is not None:
            result["mmproj_filename"] = mmproj_filename
            result["mmproj_path"] = str(mmproj_cached)
    return result


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
        lambda: snapshot_download(
            **snapshot_kwargs
        ),  # nosec B615: revision pinned in snapshot_kwargs
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
        "size_bytes": model_weight_size_bytes(root),
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
        shutil.rmtree(link)
    link.symlink_to(target, target_is_directory=target.is_dir())
    return link
