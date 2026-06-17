"""Knowledge base / RAG routes."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from forge.api.deps import get_knowledge_orchestrator
from forge.config import ForgeSettings, get_settings
from forge.orchestrators.knowledge import KnowledgeOrchestrator
from forge.security.auth import get_current_user_id
from seiso.security import SecurityError, safe_join

router = APIRouter(prefix="/knowledge", tags=["knowledge"])

_KB_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
_FILENAME_RE = re.compile(r"^[a-zA-Z0-9._-]{1,128}$")
_MAX_UPLOAD_BYTES = 50 * 1024 * 1024


class IngestRequest(BaseModel):
    knowledge_base_id: str
    source_path: str


class RetrieveRequest(BaseModel):
    knowledge_base_id: str
    query: str
    top_k: int = Field(default=5, ge=1, le=20)


class CreateBaseRequest(BaseModel):
    knowledge_base_id: str = Field(min_length=1, max_length=64)
    name: str = Field(default="", max_length=128)


def _validate_kb_id(kb_id: str) -> str:
    if not _KB_ID_RE.match(kb_id):
        raise HTTPException(400, "knowledge_base_id must be 1–64 alphanumeric characters, hyphens, or underscores")
    return kb_id


def _count_chunks(index_path: Path) -> int:
    if not index_path.exists():
        return 0
    count = 0
    with index_path.open() as f:
        for line in f:
            if line.strip():
                count += 1
    return count


@router.get("/bases")
async def list_bases(
    user_id: Annotated[str, Depends(get_current_user_id)],
    settings: Annotated[ForgeSettings, Depends(get_settings)],
) -> dict:
    try:
        kb_root = safe_join(settings.data_dir, "knowledge", user_id)
    except SecurityError as exc:
        raise HTTPException(400, str(exc)) from exc

    bases: list[dict] = []
    if kb_root.exists():
        for entry in sorted(kb_root.iterdir()):
            if not entry.is_dir():
                continue
            index_path = entry / "index.jsonl"
            bases.append(
                {
                    "id": entry.name,
                    "chunk_count": _count_chunks(index_path),
                    "has_index": index_path.exists(),
                }
            )
    return {"bases": bases}


@router.post("/bases")
async def create_base(
    body: CreateBaseRequest,
    user_id: Annotated[str, Depends(get_current_user_id)],
    settings: Annotated[ForgeSettings, Depends(get_settings)],
) -> dict:
    kb_id = _validate_kb_id(body.knowledge_base_id)
    try:
        kb_dir = safe_join(settings.data_dir, "knowledge", user_id, kb_id)
        kb_dir.mkdir(parents=True, exist_ok=True)
    except SecurityError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"id": kb_id, "name": body.name or kb_id, "path": str(kb_dir)}


@router.post("/upload")
async def upload_file(
    user_id: Annotated[str, Depends(get_current_user_id)],
    settings: Annotated[ForgeSettings, Depends(get_settings)],
    file: UploadFile = File(...),
) -> dict:
    try:
        uploads = safe_join(settings.data_dir, "uploads", user_id)
        uploads.mkdir(parents=True, exist_ok=True)
    except SecurityError as exc:
        raise HTTPException(400, str(exc)) from exc

    raw_name = Path(file.filename or "upload.txt").name
    if not _FILENAME_RE.match(raw_name):
        raise HTTPException(400, "Filename must contain only letters, numbers, dots, hyphens, or underscores")

    content = await file.read()
    if len(content) > _MAX_UPLOAD_BYTES:
        raise HTTPException(400, f"File exceeds {_MAX_UPLOAD_BYTES // (1024 * 1024)} MiB limit")

    dest = uploads / raw_name
    dest.write_bytes(content)
    return {"path": str(dest), "filename": raw_name, "size": len(content)}


@router.post("/ingest")
async def ingest(
    body: IngestRequest,
    user_id: Annotated[str, Depends(get_current_user_id)],
    orchestrator: Annotated[KnowledgeOrchestrator, Depends(get_knowledge_orchestrator)],
    settings: Annotated[ForgeSettings, Depends(get_settings)],
) -> dict:
    _validate_kb_id(body.knowledge_base_id)
    try:
        uploads = safe_join(settings.data_dir, "uploads", user_id)
        uploads.mkdir(parents=True, exist_ok=True)
    except SecurityError as exc:
        raise HTTPException(400, str(exc)) from exc

    job_id = orchestrator.create_job(user_id=user_id)
    payload = {"action": "ingest", "user_id": user_id, **body.model_dump()}
    await orchestrator.start(job_id, payload)
    job = await orchestrator.wait_for(job_id)
    if job and job.status.value == "failed":
        raise HTTPException(400, job.error or "Ingest failed")
    return {"job_id": job_id, **(job.result if job else {})}


@router.post("/retrieve")
async def retrieve(
    body: RetrieveRequest,
    user_id: Annotated[str, Depends(get_current_user_id)],
    orchestrator: Annotated[KnowledgeOrchestrator, Depends(get_knowledge_orchestrator)],
) -> dict:
    _validate_kb_id(body.knowledge_base_id)
    job_id = orchestrator.create_job(user_id=user_id)
    payload = {"action": "retrieve", "user_id": user_id, **body.model_dump()}
    await orchestrator.start(job_id, payload)
    job = await orchestrator.wait_for(job_id)
    return job.result if job else {"results": []}
