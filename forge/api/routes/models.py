"""Model hub, catalog, download, and VRAM management routes."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from forge.api.deps import get_db, get_inference_orchestrator
from forge.config import ForgeSettings, get_settings
from forge.db.store import Database
from forge.orchestrators.inference import InferenceOrchestrator
from forge.security.auth import get_current_user_id
from forge.services.model_download import perform_model_download
from forge.services.user_paths import assert_user_path
from forge.services.hardware import enrich_catalog_models, hardware_profile, hardware_summary
from forge.services.hf_auth import resolve_hf_token
from forge.services.publishable import is_pushable_model
from seiso.models.catalog import get_families, search_catalog
from seiso.security import SecurityError, sanitize_filename

router = APIRouter(prefix="/models", tags=["models"])


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
    q: str = Query("", description="Search query"),
    family: str | None = Query(None),
    task: str | None = Query(None),
    hardware_aware: bool = Query(True, description="Rank and annotate by local hardware fit"),
    fits_only: bool = Query(False, description="Show only ideal/good fits for this machine"),
) -> dict:
    models = search_catalog(q, family, task)
    profile = hardware_profile()
    hf_token: str | None = None
    if hardware_aware:
        hf_token, _ = resolve_hf_token(
            user_id=user_id,
            data_dir=settings.data_dir,
            encryption_key=settings.db_encryption_key_bytes,
            settings_token=settings.hf_token or None,
        )
        models = enrich_catalog_models(models, profile, token=hf_token)
    if fits_only:
        models = [m for m in models if m.get("hardware_fit") in ("ideal", "good")]
    return {
        "models": models,
        "families": get_families(),
        "total": len(models),
        "hardware_summary": hardware_summary(profile),
        "local_only": True,
    }


@router.get("/vram")
async def vram_status(
    user_id: Annotated[str, Depends(get_current_user_id)],
    orchestrator: Annotated[InferenceOrchestrator, Depends(get_inference_orchestrator)],
) -> dict:
    return orchestrator._runner._pool.status()


@router.post("/vram/unload")
async def unload_vram(
    user_id: Annotated[str, Depends(get_current_user_id)],
    orchestrator: Annotated[InferenceOrchestrator, Depends(get_inference_orchestrator)],
) -> dict:
    return await orchestrator._runner.cancel_and_unload()


@router.get("")
async def list_models(
    user_id: Annotated[str, Depends(get_current_user_id)],
    db: Annotated[Database, Depends(get_db)],
) -> list[dict]:
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
        raise HTTPException(403, str(exc)) from exc

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
        raise HTTPException(403, str(exc)) from exc
    if not folder.is_dir():
        raise HTTPException(400, "Path is not a directory")

    found: list[dict] = []
    for root, _, files in os.walk(folder):
        for fname in files:
            if fname.endswith((".gguf", ".safetensors", ".bin")):
                fpath = Path(root) / fname
                fmt = fpath.suffix.lstrip(".")
                entry = await db.add_model(
                    user_id=user_id,
                    name=sanitize_filename(fname),
                    path=str(fpath),
                    source="scan",
                    format=fmt,
                    size_bytes=fpath.stat().st_size,
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
            db_encryption_key=settings.db_encryption_key_bytes,
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

    def on_progress(payload: dict[str, Any]) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, ("progress", payload))

    async def run_download() -> None:
        try:
            result = await perform_model_download(
                user_id=user_id,
                db=db,
                data_dir=settings.data_dir,
                hf_cache_dir=settings.hf_cache_dir,
                settings_hf_token=settings.hf_token,
                db_encryption_key=settings.db_encryption_key_bytes,
                repo_id=body.repo_id,
                filename=body.filename,
                revision=body.revision,
                variant=body.variant,
                on_progress=on_progress,
            )
            await queue.put(("complete", result))
        except Exception as exc:
            await queue.put(("error", str(exc)))

    async def event_gen():
        task = asyncio.create_task(run_download())
        try:
            while True:
                kind, payload = await queue.get()
                if kind == "progress":
                    yield {"event": "progress", "data": json.dumps(payload)}
                elif kind == "complete":
                    yield {"event": "complete", "data": json.dumps(payload)}
                    break
                elif kind == "error":
                    yield {"event": "error", "data": str(payload)}
                    break
        finally:
            if not task.done():
                task.cancel()

    return EventSourceResponse(event_gen())


def _dir_size(path: Path) -> int:
    total = 0
    if path.is_file():
        return path.stat().st_size
    for f in path.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total


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
        raise HTTPException(403, str(exc)) from exc
    if not path.exists():
        raise HTTPException(404, "Model path not found")
    return await db.add_model(
        user_id=user_id,
        name=body.name,
        path=str(path),
        source=body.source or "manual",
        format=body.format or path.suffix.lstrip("."),
        size_bytes=path.stat().st_size if path.is_file() else _dir_size(path),
    )
