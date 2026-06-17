"""Model hub, catalog, download, and VRAM management routes."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from forge.api.deps import get_db, get_inference_orchestrator
from forge.config import ForgeSettings, get_settings
from forge.db.store import Database
from forge.orchestrators.inference import InferenceOrchestrator
from forge.security.auth import get_current_user_id
from seiso.models.catalog import get_families, search_catalog
from forge.services.user_paths import assert_user_path, user_dir
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
    q: str = Query("", description="Search query"),
    family: str | None = Query(None),
    task: str | None = Query(None),
) -> dict:
    return {
        "models": search_catalog(q, family, task),
        "families": get_families(),
        "total": len(search_catalog(q, family, task)),
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
    return await orchestrator._runner.unload()


@router.get("")
async def list_models(
    user_id: Annotated[str, Depends(get_current_user_id)],
    db: Annotated[Database, Depends(get_db)],
) -> list[dict]:
    return await db.list_models(user_id)


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
    from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download

    token = settings.hf_token or None
    dest_dir = user_dir(settings.data_dir, user_id, "models") / sanitize_filename(body.repo_id.replace("/", "--"))
    dest_dir.mkdir(parents=True, exist_ok=True)

    if body.variant == "safetensors" or (body.variant == "auto" and not body.filename):
        # Full model snapshot for training
        path = snapshot_download(
            repo_id=body.repo_id,
            local_dir=str(dest_dir),
            token=token,
            revision=body.revision,
            ignore_patterns=["*.md", "*.h5", "original/*"],
        )
        await db.add_model(
            user_id=user_id,
            name=body.repo_id.split("/")[-1],
            path=path,
            source=f"hf:{body.repo_id}",
            format="safetensors",
            size_bytes=_dir_size(Path(path)),
        )
        return {"downloaded": [path], "repo_id": body.repo_id, "variant": "safetensors"}

    if body.filename:
        files = [body.filename]
    else:
        files = [
            f
            for f in list_repo_files(body.repo_id, token=token)
            if f.endswith((".gguf", ".safetensors", ".bin"))
        ][:5]

    downloaded: list[str] = []
    for fname in files:
        path = hf_hub_download(
            repo_id=body.repo_id,
            filename=fname,
            revision=body.revision,
            local_dir=str(dest_dir),
            token=token,
        )
        downloaded.append(path)
        await db.add_model(
            user_id=user_id,
            name=fname,
            path=path,
            source=f"hf:{body.repo_id}",
            format=Path(fname).suffix.lstrip("."),
            size_bytes=Path(path).stat().st_size,
        )

    return {"downloaded": downloaded, "repo_id": body.repo_id}


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
