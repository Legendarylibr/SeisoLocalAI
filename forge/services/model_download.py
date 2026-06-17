"""Shared model download orchestration for sync and streaming routes."""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from typing import Any

from forge.db.store import Database
from forge.services.download_progress import ProgressCallback
from forge.services.hf_auth import resolve_hf_token_for_download
from forge.services.hf_connectivity import assert_hub_ready_for_download, check_inference_runtime
from forge.services.hf_hub import (
    download_gguf,
    download_training_snapshot,
    link_inventory,
    resolve_gguf_artifact,
)
from forge.services.user_paths import user_dir
from seiso.models.catalog import get_by_repo
from seiso.security import sanitize_filename


def _emit_progress(on_progress: ProgressCallback | None, payload: dict[str, Any]) -> None:
    if on_progress:
        on_progress(payload)


def _format_gib(size_bytes: int) -> str:
    return f"{size_bytes / (1024 ** 3):.1f} GB"


def _assert_disk_space_for_download(cache_dir: Path, total_bytes: int) -> None:
    """Fail early when the target cache filesystem cannot hold the resolved artifact."""
    if total_bytes <= 0:
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(cache_dir).free
    # Hugging Face may need temp/partial files while resuming, so keep a small cushion.
    required_bytes = int(total_bytes * 1.05)
    if free_bytes < required_bytes:
        raise ValueError(
            "Not enough free disk space for model download. "
            f"Need about {_format_gib(required_bytes)} in {cache_dir}, "
            f"but only {_format_gib(free_bytes)} is free."
        )


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
    if variant == "auto":
        runtime = check_inference_runtime()
        if runtime.llamacpp:
            variant = "gguf"
        elif runtime.mlx or runtime.torch:
            variant = "safetensors"
        else:
            variant = "gguf"
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
        entry=entry,
        inventory_repo_id=catalog_repo,
        on_progress=on_progress,
        total_bytes=total_bytes if total_bytes > 0 else None,
    )
    cached = Path(info["path"])
    inv = link_inventory(inventory_dir, info["inventory_name"], cached)
    return {
        "variant": "gguf",
        "source": source,
        "name": cached.name,
        "path": str(inv.absolute()),
        "format": "gguf",
        "size_bytes": cached.stat().st_size,
        "metadata": {
            "repo_id": catalog_repo,
            "gguf_repo": gguf_repo,
            "cache_dir": str(cache_dir),
            "gguf_file": info["filename"],
        },
        "downloaded": [info["path"]],
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
