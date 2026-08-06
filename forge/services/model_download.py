"""Shared model download orchestration for sync and streaming routes."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import threading
from pathlib import Path
from typing import Any

from forge.db.store import Database
from forge.services.artifact_integrity import (
    gguf_files_complete_at_path,
    path_has_complete_artifact,
)
from forge.services.download_progress import ProgressCallback
from forge.services.hf_auth import resolve_hf_token_for_download
from forge.services.hf_connectivity import assert_hub_ready_for_download, check_inference_runtime
from forge.services.hf_hub import (
    download_gguf,
    download_training_snapshot,
    estimate_snapshot_download_bytes,
    get_gguf_file_size_bytes,
    link_inventory,
    resolve_gguf_artifact,
)
from forge.services.model_download_worker import sync_download_artifacts
from seiso.io.files import iter_matching_files, path_size_bytes
from seiso.models.catalog import get_by_repo

__all__ = [
    "_assert_disk_space_for_download",
    "_sync_download_artifacts",
    "assert_hub_ready_for_download",
    "download_gguf",
    "download_training_snapshot",
    "estimate_snapshot_download_bytes",
    "find_inventory_for_catalog_repo",
    "get_by_repo",
    "link_inventory",
    "perform_model_download",
    "path_size_bytes",
    "resolve_download_variant",
    "resolve_gguf_artifact",
    "resolve_hf_token_for_download",
]

_DOWNLOAD_LOCKS: dict[str, asyncio.Lock] = {}
_DOWNLOAD_LOCKS_GUARD = threading.Lock()


def _emit_progress(on_progress: ProgressCallback | None, payload: dict[str, Any]) -> None:
    if on_progress:
        on_progress(payload)


def resolve_download_variant(variant: str) -> str:
    """Resolve ``auto`` to ``gguf`` or ``safetensors`` based on local inference runtime."""
    if variant != "auto":
        return variant
    runtime = check_inference_runtime()
    if runtime.llamacpp or getattr(runtime, "llamaswap", False):
        return "gguf"
    if runtime.mlx or runtime.torch:
        return "safetensors"
    return "gguf"


def _format_gib(size_bytes: int) -> str:
    return f"{size_bytes / (1024**3):.1f} GB"


def _assert_disk_space_for_download(cache_dir: Path, total_bytes: int) -> None:
    """Fail early when the target cache filesystem cannot hold the resolved artifact."""
    if total_bytes <= 0:
        return
    # Hugging Face may need temp/partial files while resuming, so keep a small cushion.
    required_bytes = int(total_bytes * 1.05)
    checked: set[Path] = set()
    candidates = [cache_dir]
    if raw_xet_cache := os.environ.get("HF_XET_CACHE"):
        candidates.append(Path(raw_xet_cache).expanduser())
    for target in candidates:
        target.mkdir(parents=True, exist_ok=True)
        resolved = target.resolve()
        if resolved in checked:
            continue
        checked.add(resolved)
        free_bytes = shutil.disk_usage(resolved).free
        if free_bytes < required_bytes:
            raise ValueError(
                "Not enough free disk space for model download. "
                f"Need about {_format_gib(required_bytes)} in {resolved}, "
                f"but only {_format_gib(free_bytes)} is free."
            )


def _download_lock_key(
    *,
    user_id: str,
    repo_id: str,
    filename: str | None,
    revision: str,
    variant: str,
) -> str:
    return "\0".join((user_id, repo_id, filename or "", revision, variant))


def _get_download_lock(key: str) -> asyncio.Lock:
    with _DOWNLOAD_LOCKS_GUARD:
        return _DOWNLOAD_LOCKS.setdefault(key, asyncio.Lock())


async def find_inventory_for_catalog_repo(
    db: Database,
    user_id: str,
    catalog_repo: str,
) -> dict[str, Any] | None:
    """Find a local inventory row for a catalog repo id (handles GGUF mirror sources)."""
    existing = await db.get_model_by_source(user_id, f"hf:{catalog_repo}")
    if existing:
        return existing
    by_meta = await db.get_model_by_metadata_repo_id(user_id, catalog_repo)
    if by_meta:
        return by_meta
    return None


def _cached_download_result_if_usable(
    existing: dict[str, Any] | None,
    *,
    repo_id: str,
    variant: str,
) -> dict[str, Any] | None:
    if not existing:
        return None
    path = Path(str(existing.get("path") or ""))
    fmt = str(existing.get("format") or "").lower()
    expected_size = int(existing.get("size_bytes") or 0)
    try:
        metadata = json.loads(existing.get("metadata_json") or "{}")
    except json.JSONDecodeError:
        metadata = {}
    exact_gguf_files_complete = False
    if fmt == "gguf":
        gguf_repo = str(metadata.get("gguf_repo") or metadata.get("repo_id") or repo_id)
        gguf_files = metadata.get("gguf_files") or metadata.get("gguf_file")
        if isinstance(gguf_files, str):
            gguf_files = [gguf_files]
        if not isinstance(gguf_files, list) or not gguf_files:
            if str(existing.get("source") or "").startswith("hf:"):
                return None
        else:
            with contextlib.suppress(Exception):
                expected_size = max(
                    expected_size,
                    sum(get_gguf_file_size_bytes(gguf_repo, str(item)) for item in gguf_files),
                )
            if not gguf_files_complete_at_path(
                path, [str(item) for item in gguf_files], expected_size
            ):
                return None
            exact_gguf_files_complete = True
    if not exact_gguf_files_complete and not path_has_complete_artifact(path, fmt, expected_size):
        return None
    requested = variant.lower()
    if requested == "gguf" and fmt != "gguf":
        return None
    if requested == "safetensors" and fmt == "gguf":
        return None
    cached_variant = "gguf" if fmt == "gguf" else "safetensors"
    downloaded = [str(path)]
    if fmt == "gguf" and path.is_dir():
        if isinstance(gguf_files, list) and gguf_files:
            downloaded = [
                str(path / str(filename))
                for filename in gguf_files
                if (path / str(filename)).is_file()
            ]
        else:
            downloaded = [str(p) for p in iter_matching_files(path, "*.gguf")]
    return {
        "downloaded": downloaded,
        "repo_id": repo_id,
        "variant": cached_variant,
        "model_id": existing["id"],
        "cache_dir": metadata.get("cache_dir"),
        **({"gguf_repo": metadata["gguf_repo"]} if metadata.get("gguf_repo") else {}),
        "cached": True,
    }


async def perform_model_download(
    *,
    user_id: str,
    db: Database,
    data_dir: Path,
    hf_cache_dir: Path,
    settings_hf_token: str | None,
    db_encryption_key: bytes,
    repo_id: str,
    filename: str | None = None,
    revision: str = "main",
    variant: str = "auto",
    on_progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    resolved_variant = resolve_download_variant(variant)
    key = _download_lock_key(
        user_id=user_id,
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        variant=resolved_variant,
    )
    lock = _get_download_lock(key)
    if lock.locked():
        _emit_progress(
            on_progress,
            {
                "phase": "resolving",
                "label": f"Waiting for existing Hugging Face download of {repo_id}",
                "repo_id": repo_id,
                "percent": 0,
            },
        )
    async with lock:
        existing = await find_inventory_for_catalog_repo(db, user_id, repo_id)
        cached = _cached_download_result_if_usable(
            existing,
            repo_id=repo_id,
            variant=resolved_variant,
        )
        if cached:
            _emit_progress(
                on_progress,
                {
                    "phase": "finalizing",
                    "label": f"Using cached Hugging Face model for {repo_id}",
                    "repo_id": repo_id,
                    "percent": 100,
                },
            )
            return cached

        from forge.services.memory_release import prepare_for_gpu_task, release_after_task

        loop = asyncio.get_running_loop()
        prep = await loop.run_in_executor(
            None,
            lambda: prepare_for_gpu_task(task="download", user_id=user_id),
        )
        _emit_progress(
            on_progress,
            {
                "phase": "resolving",
                "label": f"Released inference memory for {repo_id} download",
                "repo_id": repo_id,
                "percent": 2,
            },
        )

        try:
            artifacts = await loop.run_in_executor(
                None,
                lambda: _sync_download_artifacts(
                    catalog_repo=repo_id,
                    data_dir=data_dir,
                    hf_cache_dir=hf_cache_dir,
                    settings_hf_token=settings_hf_token,
                    db_encryption_key=db_encryption_key,
                    user_id=user_id,
                    filename=filename,
                    revision=revision,
                    variant=variant,
                    on_progress=on_progress,
                ),
            )
        finally:
            await loop.run_in_executor(
                None,
                lambda: release_after_task(
                    reason="download complete",
                    resource_token=str(prep.get("resource_token") or ""),
                ),
            )
        _emit_progress(
            on_progress,
            {
                "phase": "finalizing",
                "label": f"Registering {artifacts['name']} in local model inventory",
                "repo_id": repo_id,
                "percent": 99,
            },
        )
        record = await db.upsert_model(
            user_id=user_id,
            source=artifacts["source"],
            name=artifacts["name"],
            path=artifacts["path"],
            format=artifacts["format"],
            size_bytes=artifacts["size_bytes"],
            metadata=artifacts["metadata"],
        )
        return {
            "downloaded": artifacts["downloaded"],
            "repo_id": artifacts["repo_id"],
            "variant": artifacts["variant"],
            "model_id": record["id"],
            "cache_dir": artifacts["cache_dir"],
            **({"gguf_repo": artifacts["gguf_repo"]} if "gguf_repo" in artifacts else {}),
        }


# Backward-compatible alias for tests and callers that patch the worker.
_sync_download_artifacts = sync_download_artifacts
