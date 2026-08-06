"""Blocking Hugging Face model download worker — safe to run in a thread pool."""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

from forge.services.download_progress import ProgressCallback
from forge.services.user_paths import user_dir
from seiso.security import sanitize_filename


def _emit_progress(on_progress: ProgressCallback | None, payload: dict[str, Any]) -> None:
    if on_progress:
        on_progress(payload)


def _maybe_register_with_ollama(
    model_path: str,
    *,
    catalog_repo: str,
    metadata: dict[str, Any],
    model_format: str,
) -> None:
    try:
        from forge.services.ollama_registry import register_model_with_ollama

        metadata["ollama_tag"] = register_model_with_ollama(
            str(model_path),
            repo_id=catalog_repo,
            metadata=metadata,
            model_format=model_format,
        )
    except ValueError:
        return
    except Exception:
        return


def sync_download_artifacts(
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
    from forge.services import model_download as md

    resolved_variant = md.resolve_download_variant(variant)
    use_safetensors = resolved_variant == "safetensors"
    md.assert_hub_ready_for_download(
        user_id=user_id,
        data_dir=data_dir,
        encryption_key=db_encryption_key,
        settings_token=settings_hf_token or None,
    )
    token, _ = md.resolve_hf_token_for_download(
        user_id=user_id,
        data_dir=data_dir,
        encryption_key=db_encryption_key,
        settings_token=settings_hf_token or None,
    )
    cache_dir = hf_cache_dir
    inventory_dir = user_dir(data_dir, user_id, "models")
    source = f"hf:{catalog_repo}"

    if use_safetensors:
        _emit_progress(
            on_progress,
            {
                "phase": "resolving",
                "label": f"Preparing training snapshot for {catalog_repo}",
                "percent": 0,
            },
        )
        try:
            snapshot_bytes = md.estimate_snapshot_download_bytes(
                catalog_repo,
                token=token,
                revision=revision,
            )
        except Exception:
            snapshot_bytes = 0
        md._assert_disk_space_for_download(cache_dir, snapshot_bytes)
        info = md.download_training_snapshot(
            catalog_repo,
            cache_dir=cache_dir,
            token=token,
            revision=revision,
            on_progress=on_progress,
        )
        inv = md.link_inventory(
            inventory_dir,
            sanitize_filename(catalog_repo.replace("/", "--")),
            Path(info["path"]),
        )
        metadata: dict[str, Any] = {"repo_id": catalog_repo, "cache_dir": str(cache_dir)}
        _maybe_register_with_ollama(
            str(inv.absolute()),
            catalog_repo=catalog_repo,
            metadata=metadata,
            model_format="safetensors",
        )
        return {
            "variant": "safetensors",
            "source": source,
            "name": catalog_repo.split("/")[-1],
            "path": str(inv.absolute()),
            "format": "safetensors",
            "size_bytes": info["size_bytes"],
            "metadata": metadata,
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
    entry = md.get_by_repo(catalog_repo, token=token)
    artifact = md.resolve_gguf_artifact(
        catalog_repo,
        token=token,
        revision=revision,
        entry=entry,
        filename=filename,
    )
    gguf_repo = artifact["gguf_repo"]
    gguf_file = artifact["filename"]
    gguf_files = list(artifact.get("filenames") or [gguf_file])
    mmproj_file = artifact.get("mmproj_filename")
    total_bytes = int(artifact.get("size_bytes") or 0)
    md._assert_disk_space_for_download(cache_dir, total_bytes)
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
    info = md.download_gguf(
        gguf_repo,
        cache_dir=cache_dir,
        token=token,
        revision=revision,
        filename=gguf_file,
        filenames=gguf_files,
        mmproj_filename=mmproj_file,
        entry=entry,
        inventory_repo_id=catalog_repo,
        on_progress=on_progress,
        total_bytes=total_bytes if total_bytes > 0 else None,
    )
    cached = Path(info["path"])
    inv = md.link_inventory(inventory_dir, info["inventory_name"], cached)
    if info.get("mmproj_path") and info.get("mmproj_filename"):
        inv_parent = Path(info["inventory_name"]).parent
        mmproj_inventory = str(inv_parent / Path(info["mmproj_filename"]).name)
        md.link_inventory(inventory_dir, mmproj_inventory, Path(info["mmproj_path"]))
    downloaded_paths = [Path(raw) for raw in info.get("paths") or [info["path"]]]
    try:
        size_bytes = sum(path.stat().st_size for path in downloaded_paths)
    except OSError:
        size_bytes = md.path_size_bytes(cached)
    metadata = {
        "repo_id": catalog_repo,
        "gguf_repo": gguf_repo,
        "cache_dir": str(cache_dir),
        "gguf_file": info["filename"],
        "gguf_files": info.get("filenames") or [info["filename"]],
        **({"mmproj_file": info["mmproj_filename"]} if info.get("mmproj_filename") else {}),
    }
    with contextlib.suppress(Exception):
        _maybe_register_with_ollama(
            str(inv.absolute()),
            catalog_repo=catalog_repo,
            metadata=metadata,
            model_format="gguf",
        )
    return {
        "variant": "gguf",
        "source": source,
        "name": cached.name if cached.is_file() else Path(info["filename"]).stem,
        "path": str(inv.absolute()),
        "format": "gguf",
        "size_bytes": size_bytes,
        "metadata": metadata,
        "downloaded": [str(path) for path in downloaded_paths],
        "repo_id": catalog_repo,
        "gguf_repo": gguf_repo,
        "cache_dir": str(cache_dir),
    }
