"""Model hub, catalog, download, and VRAM management routes."""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from forge.api.deps import get_db, get_inference_orchestrator
from forge.api.http_errors import raise_forbidden
from forge.config import ForgeSettings, get_settings
from forge.db.store import Database
from forge.orchestrators.inference import InferenceOrchestrator
from forge.security.auth import get_current_user_id
from forge.services.hardware import enrich_catalog_models, hardware_profile, hardware_summary
from forge.services.hf_auth import resolve_hf_token
from forge.services.hf_cache_inventory import sync_hf_cache_inventory
from forge.services.hf_hub import _format_hub_download_error, dir_size
from forge.services.model_download import perform_model_download
from forge.services.publishable import is_pushable_model
from forge.services.user_paths import assert_user_path
from seiso.models.catalog import HubSearchError, get_families, search_catalog
from seiso.security import SecurityError, sanitize_filename

router = APIRouter(prefix="/models", tags=["models"])


class DownloadStreamClosed(RuntimeError):
    """Raised inside the download worker when the SSE client disconnects."""


class ModelScanRequest(BaseModel):
    path: str


class ModelDownloadRequest(BaseModel):
    repo_id: str
    filename: str | None = None
    revision: str = "main"
    variant: str = Field(default="auto", description="auto | safetensors | gguf")


class LocalModelCreate(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    path: str
    source: str | None = None
    format: str | None = None


@router.get("/catalog")
async def model_catalog(
    user_id: Annotated[str, Depends(get_current_user_id)],
    settings: Annotated[ForgeSettings, Depends(get_settings)],
    q: str = Query("", description="Search Hugging Face Hub"),
    family: str | None = Query(None),
    task: str | None = Query(None),
    hardware_aware: bool = Query(True, description="Rank and annotate by local hardware fit"),
    fits_only: bool = Query(False, description="Show only ideal/good fits for this machine"),
    limit: int = Query(50, ge=1, le=100),
    cursor: str | None = Query(None, description="Hugging Face Hub pagination cursor"),
) -> dict:
    hf_token, _ = resolve_hf_token(
        user_id=user_id,
        data_dir=settings.data_dir,
        encryption_key=settings.hf_token_encryption_key,
        settings_token=settings.hf_token or None,
    )
    try:
        result = search_catalog(
            q,
            family,
            task,
            limit=limit,
            cursor=cursor,
            token=hf_token,
        )
    except HubSearchError as exc:
        status = 429 if exc.status_code == 429 else 502
        raise HTTPException(status, str(exc)) from exc

    models = result.models
    profile = hardware_profile()
    if hardware_aware:
        models = enrich_catalog_models(
            models,
            profile,
            token=hf_token,
            diversify=False,
        )
    if fits_only:
        models = [m for m in models if m.get("hardware_fit") in ("ideal", "good")]
    return {
        "models": models,
        "families": get_families(),
        "total": len(models),
        "limit": limit,
        "next_cursor": result.next_cursor,
        "has_more": result.next_cursor is not None,
        "source": "huggingface",
        "hardware_summary": hardware_summary(profile),
        "local_only": True,
    }


@router.get("/vram")
async def vram_status(
    user_id: Annotated[str, Depends(get_current_user_id)],
    orchestrator: Annotated[InferenceOrchestrator, Depends(get_inference_orchestrator)],
) -> dict:
    from forge.services.hardware import build_vram_status

    return build_vram_status(orchestrator)


@router.post("/vram/unload")
async def unload_vram(
    user_id: Annotated[str, Depends(get_current_user_id)],
    orchestrator: Annotated[InferenceOrchestrator, Depends(get_inference_orchestrator)],
) -> dict:
    try:
        return await orchestrator.release_all_inference_memory(user_id)
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc


@router.get("")
async def list_models(
    user_id: Annotated[str, Depends(get_current_user_id)],
    db: Annotated[Database, Depends(get_db)],
    settings: Annotated[ForgeSettings, Depends(get_settings)],
) -> list[dict]:
    await sync_hf_cache_inventory(
        db,
        user_id,
        data_dir=settings.data_dir,
        hf_cache_dir=settings.hf_cache_dir,
    )
    models = await db.list_models(user_id)
    for m in models:
        m["pushable"] = is_pushable_model(m)
    return models


@router.get("/{model_id}/download")
async def download_local_model(
    model_id: str,
    user_id: Annotated[str, Depends(get_current_user_id)],
    db: Annotated[Database, Depends(get_db)],
    settings: Annotated[ForgeSettings, Depends(get_settings)],
):
    """Download a local GGUF or other model file to the browser."""
    model = await db.get_model(model_id, user_id)
    if not model:
        raise HTTPException(404, "Model not found")
    try:
        path = assert_user_path(settings.data_dir, user_id, model["path"])
    except SecurityError as exc:
        raise_forbidden(exc)

    if path.is_dir():
        ggufs = sorted(path.glob("*.gguf"))
        if ggufs:
            path = ggufs[0]
        else:
            raise HTTPException(404, "No downloadable file in model directory")

    if not path.is_file():
        raise HTTPException(404, "File not found")

    return FileResponse(path, filename=path.name, media_type="application/octet-stream")


@router.post("/scan")
async def scan_folder(
    body: ModelScanRequest,
    user_id: Annotated[str, Depends(get_current_user_id)],
    db: Annotated[Database, Depends(get_db)],
    settings: Annotated[ForgeSettings, Depends(get_settings)],
) -> list[dict]:
    try:
        folder = assert_user_path(settings.data_dir, user_id, body.path)
    except SecurityError as exc:
        raise_forbidden(exc)
    if not folder.is_dir():
        raise HTTPException(400, "Path is not a directory")

    found: list[dict] = []
    for root, _, files in os.walk(folder, followlinks=False):
        for fname in files:
            if not fname.endswith((".gguf", ".safetensors", ".bin")):
                continue
            fpath = Path(root) / fname
            if fpath.is_symlink():
                continue
            try:
                validated = assert_user_path(settings.data_dir, user_id, fpath)
            except SecurityError:
                continue
            fmt = validated.suffix.lstrip(".")
            entry = await db.add_model(
                user_id=user_id,
                name=sanitize_filename(validated.name),
                path=str(validated),
                source="scan",
                format=fmt,
                size_bytes=validated.stat().st_size,
            )
            found.append(entry)
    return found


@router.post("/download")
async def download_model(
    body: ModelDownloadRequest,
    user_id: Annotated[str, Depends(get_current_user_id)],
    db: Annotated[Database, Depends(get_db)],
    settings: Annotated[ForgeSettings, Depends(get_settings)],
) -> dict[str, Any]:
    try:
        return await perform_model_download(
            user_id=user_id,
            db=db,
            data_dir=settings.data_dir,
            hf_cache_dir=settings.hf_cache_dir,
            settings_hf_token=settings.hf_token,
            db_encryption_key=settings.hf_token_encryption_key,
            repo_id=body.repo_id,
            filename=body.filename,
            revision=body.revision,
            variant=body.variant,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/download/stream")
async def download_model_stream(
    body: ModelDownloadRequest,
    user_id: Annotated[str, Depends(get_current_user_id)],
    db: Annotated[Database, Depends(get_db)],
    settings: Annotated[ForgeSettings, Depends(get_settings)],
):
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
    stream_open = True

    def on_progress(payload: dict[str, Any]) -> None:
        if not stream_open:
            return
        loop.call_soon_threadsafe(queue.put_nowait, ("progress", payload))

    async def run_download() -> None:
        try:
            result = await perform_model_download(
                user_id=user_id,
                db=db,
                data_dir=settings.data_dir,
                hf_cache_dir=settings.hf_cache_dir,
                settings_hf_token=settings.hf_token,
                db_encryption_key=settings.hf_token_encryption_key,
                repo_id=body.repo_id,
                filename=body.filename,
                revision=body.revision,
                variant=body.variant,
                on_progress=on_progress,
            )
            if stream_open:
                await queue.put(("complete", result))
        except DownloadStreamClosed:
            return
        except Exception as exc:
            if stream_open:
                msg = (
                    str(exc)
                    if isinstance(exc, ValueError)
                    else _format_hub_download_error(exc, repo_id=body.repo_id)
                )
                await queue.put(("error", msg))

    async def event_gen():
        nonlocal stream_open
        download_task = asyncio.create_task(run_download())
        _ = download_task  # silence ruff F841; task is awaited/cancelled below
        started_at = time.monotonic()
        last_progress: dict[str, Any] | None = None
        try:
            while True:
                try:
                    kind, payload = await asyncio.wait_for(queue.get(), timeout=2.0)
                except asyncio.TimeoutError:
                    elapsed = time.monotonic() - started_at
                    if last_progress is not None:
                        heartbeat = dict(last_progress)
                        heartbeat["heartbeat"] = True
                        heartbeat["elapsed_seconds"] = int(elapsed)
                        if (
                            int(heartbeat.get("bytes") or 0) <= 0
                            and heartbeat.get("phase") == "download"
                        ):
                            heartbeat["label"] = (
                                heartbeat.get("label")
                                or f"Downloading {body.repo_id} from Hugging Face"
                            )
                            heartbeat["waiting_for_first_byte"] = True
                        yield {"event": "progress", "data": json.dumps(heartbeat)}
                    elif elapsed >= 4:
                        yield {
                            "event": "progress",
                            "data": json.dumps(
                                {
                                    "phase": "resolving",
                                    "label": f"Resolving Hugging Face download for {body.repo_id}",
                                    "repo_id": body.repo_id,
                                    "percent": 0,
                                    "elapsed_seconds": int(elapsed),
                                    "heartbeat": True,
                                }
                            ),
                        }
                    continue
                if kind == "progress":
                    if isinstance(payload, dict):
                        last_progress = payload
                    yield {"event": "progress", "data": json.dumps(payload)}
                elif kind == "complete":
                    yield {"event": "complete", "data": json.dumps(payload)}
                    break
                elif kind == "error":
                    yield {"event": "error", "data": str(payload)}
                    break
        finally:
            stream_open = False
            # The browser may navigate away while huggingface_hub is still writing
            # the model into the shared cache. Let that worker finish so a transient
            # SSE disconnect does not corrupt or abort the download.

    return EventSourceResponse(event_gen())


@router.post("/local")
async def register_local(
    body: LocalModelCreate,
    user_id: Annotated[str, Depends(get_current_user_id)],
    db: Annotated[Database, Depends(get_db)],
    settings: Annotated[ForgeSettings, Depends(get_settings)],
) -> dict:
    try:
        path = assert_user_path(settings.data_dir, user_id, body.path)
    except SecurityError as exc:
        raise_forbidden(exc)
    if not path.exists():
        raise HTTPException(404, "Model path not found")
    return await db.add_model(
        user_id=user_id,
        name=body.name,
        path=str(path),
        source=body.source or "manual",
        format=body.format or path.suffix.lstrip("."),
        size_bytes=path.stat().st_size if path.is_file() else dir_size(path),
    )
