"""Shared model download orchestration for sync and streaming routes."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from pathlib import Path
from typing import Any

from forge.db.store import Database
from forge.services.download_progress import ProgressCallback
from forge.services.hf_auth import resolve_hf_token_for_download
from forge.services.hf_connectivity import assert_hub_ready_for_download, check_inference_runtime
from forge.services.hf_hub import (
    dir_size,
    download_gguf,
    download_training_snapshot,
    estimate_snapshot_download_bytes,
    link_inventory,
    resolve_gguf_artifact,
)
from forge.services.user_paths import user_dir
from seiso.models.catalog import get_by_repo
from seiso.security import sanitize_filename

_DOWNLOAD_LOCKS: dict[str, asyncio.Lock] = {}


def _emit_progress(on_progress: ProgressCallback | None, payload: dict[str, Any]) -> None:
    if on_progress:
        on_progress(payload)


def resolve_download_variant(variant: str) -> str:
    """Resolve ``auto`` to ``gguf`` or ``safetensors`` based on local inference runtime."""
    if variant != "auto":
        return variant
    runtime = check_inference_runtime()
    if runtime.llamacpp:
        return "gguf"
    if runtime.mlx or runtime.torch:
        return "safetensors"
    return "gguf"


def _format_gib(size_bytes: int) -> str:
    return f"{size_bytes / (1024 ** 3):.1f} GB"


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
    lock = _DOWNLOAD_LOCKS.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _DOWNLOAD_LOCKS[key] = lock
    return lock


async def find_inventory_for_catalog_repo(
    db: Database,
    user_id: str,
    catalog_repo: str,
) -> dict[str, Any] | None:
    """Find a local inventory row for a catalog repo id (handles GGUF mirror sources)."""
    existing = await db.get_model_by_source(user_id, f"hf:{catalog_repo}")
    if existing:
        return existing
    for row in await db.list_models(user_id):
        if row.get("source") == f"hf:{catalog_repo}":
            return row
        try:
            metadata = json.loads(row.get("metadata_json") or "{}")
        except json.JSONDecodeError:
            metadata = {}
        if metadata.get("repo_id") == catalog_repo:
            return row
    return None


def _path_has_complete_artifact(path: Path, fmt: str, expected_size: int) -> bool:
    if not path.exists():
        return False
    if path.name.endswith((".incomplete", ".partial", ".lock")):
        return False
    if fmt == "gguf":
        if path.is_file():
            size = path.stat().st_size
            return path.suffix.lower() == ".gguf" and size > 0 and (expected_size <= 0 or size >= expected_size)
        ggufs = [p for p in path.rglob("*.gguf") if p.is_file()]
        size = sum(p.stat().st_size for p in ggufs)
        return bool(ggufs) and size > 0 and (expected_size <= 0 or size >= expected_size)
    if path.is_dir():
        weight_files = [
            p
            for p in path.rglob("*")
            if p.is_file() and p.suffix.lower() in {".safetensors", ".bin"}
        ]
        size = sum(p.stat().st_size for p in weight_files)
        return bool(weight_files) and size > 0 and (expected_size <= 0 or size >= expected_size)
    return path.is_file() and path.stat().st_size > 0 and (expected_size <= 0 or path.stat().st_size >= expected_size)


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
    if not _path_has_complete_artifact(path, fmt, expected_size):
        return None
    requested = variant.lower()
    if requested == "gguf" and fmt != "gguf":
        return None
    if requested == "safetensors" and fmt == "gguf":
        return None
    try:
        metadata = json.loads(existing.get("metadata_json") or "{}")
    except json.JSONDecodeError:
        metadata = {}
    cached_variant = "gguf" if fmt == "gguf" else "safetensors"
    downloaded = [str(path)]
    if fmt == "gguf" and path.is_dir():
        downloaded = [str(p) for p in sorted(path.rglob("*.gguf")) if p.is_file()]
    return {
        "downloaded": downloaded,
        "repo_id": repo_id,
        "variant": cached_variant,
        "model_id": existing["id"],
        "cache_dir": metadata.get("cache_dir"),
        **({"gguf_repo": metadata["gguf_repo"]} if metadata.get("gguf_repo") else {}),
        "cached": True,
    }


def _sync_download_artifacts(
    *,
    catalog_repo: str,
    data_dir: Path,
    hf_cache_dir: Path,
    settings_hf_token: str | None,
    db_encryption_key: bytes,
    user_id: str,
    filename: str | None = None,
    revision: str = "main",
    variant: str = "auto",
    on_progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Blocking Hugging Face download — safe to run in a thread pool."""
    variant = resolve_download_variant(variant)
    assert_hub_ready_for_download(
        user_id=user_id,
        data_dir=data_dir,
        encryption_key=db_encryption_key,
        settings_token=settings_hf_token or None,
    )
    token, _ = resolve_hf_token_for_download(
        user_id=user_id,
        data_dir=data_dir,
        encryption_key=db_encryption_key,
        settings_token=settings_hf_token or None,
    )
    cache_dir = hf_cache_dir
    inventory_dir = user_dir(data_dir, user_id, "models")
    entry = get_by_repo(catalog_repo)
    source = f"hf:{catalog_repo}"

    if variant == "safetensors":
        _emit_progress(
            on_progress,
            {
                "phase": "resolving",
                "label": f"Preparing training snapshot for {catalog_repo}",
                "percent": 0,
            },
        )
        try:
            snapshot_bytes = estimate_snapshot_download_bytes(
                catalog_repo,
                token=token,
                revision=revision,
            )
        except Exception:
            snapshot_bytes = 0
        _assert_disk_space_for_download(cache_dir, snapshot_bytes)
        info = download_training_snapshot(
            catalog_repo,
            cache_dir=cache_dir,
            token=token,
            revision=revision,
            on_progress=on_progress,
        )
        inv = link_inventory(
            inventory_dir,
            sanitize_filename(catalog_repo.replace("/", "--")),
            Path(info["path"]),
        )
        return {
            "variant": "safetensors",
            "source": source,
            "name": catalog_repo.split("/")[-1],
            "path": str(inv.absolute()),
            "format": "safetensors",
            "size_bytes": info["size_bytes"],
            "metadata": {"repo_id": catalog_repo, "cache_dir": str(cache_dir)},
            "downloaded": [info["path"]],
            "repo_id": catalog_repo,
            "cache_dir": str(cache_dir),
        }

    _emit_progress(
        on_progress,
        {
            "phase": "resolving",
            "label": f"Finding GGUF quant for {catalog_repo}",
            "percent": 0,
        },
    )
    artifact = resolve_gguf_artifact(
        catalog_repo,
        token=token,
        revision=revision,
        entry=entry,
        filename=filename,
    )
    gguf_repo = artifact["gguf_repo"]
    gguf_file = artifact["filename"]
    gguf_files = list(artifact.get("filenames") or [gguf_file])
    total_bytes = int(artifact.get("size_bytes") or 0)
    _assert_disk_space_for_download(cache_dir, total_bytes)
    initial_eta = int(total_bytes / (8 * 1024 * 1024)) if total_bytes > 0 else None
    _emit_progress(
        on_progress,
        {
            "phase": "download",
            "label": f"Downloading {gguf_file} from {gguf_repo}",
            "repo_id": gguf_repo,
            "total_bytes": total_bytes,
            "bytes": 0,
            "percent": 0,
            "eta_seconds": initial_eta,
            "speed_bps": 0,
        },
    )
    info = download_gguf(
        gguf_repo,
        cache_dir=cache_dir,
        token=token,
        revision=revision,
        filename=gguf_file,
        filenames=gguf_files,
        entry=entry,
        inventory_repo_id=catalog_repo,
        on_progress=on_progress,
        total_bytes=total_bytes if total_bytes > 0 else None,
    )
    cached = Path(info["path"])
    inv = link_inventory(inventory_dir, info["inventory_name"], cached)
    size_bytes = cached.stat().st_size if cached.is_file() else dir_size(cached)
    return {
        "variant": "gguf",
        "source": source,
        "name": cached.name if cached.is_file() else Path(info["filename"]).stem,
        "path": str(inv.absolute()),
        "format": "gguf",
        "size_bytes": size_bytes,
        "metadata": {
            "repo_id": catalog_repo,
            "gguf_repo": gguf_repo,
            "cache_dir": str(cache_dir),
            "gguf_file": info["filename"],
            "gguf_files": info.get("filenames") or [info["filename"]],
        },
        "downloaded": info.get("paths") or [info["path"]],
        "repo_id": catalog_repo,
        "gguf_repo": gguf_repo,
        "cache_dir": str(cache_dir),
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

        loop = asyncio.get_running_loop()
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
