"""Knowledge base / RAG routes."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from forge.api.deps import get_knowledge_orchestrator
from forge.config import ForgeSettings, get_settings
from forge.orchestrators.knowledge import KnowledgeOrchestrator
from forge.security.auth import get_current_user_id
from forge.services.knowledge_paths import validate_kb_id
from seiso.security import SecurityError, safe_join

router = APIRouter(prefix="/knowledge", tags=["knowledge"])
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


@router.get("/bases")
async def list_bases(
    user_id: Annotated[str, Depends(get_current_user_id)],
    settings: Annotated[ForgeSettings, Depends(get_settings)],
) -> dict:
    from forge.services.knowledge_context import count_knowledge_chunks

    try:
        kb_root = safe_join(settings.data_dir, "knowledge", user_id)
    except SecurityError as exc:
        raise HTTPException(400, str(exc)) from exc

    def _scan() -> list[dict]:
        bases: list[dict] = []
        if kb_root.exists():
            for entry in sorted(kb_root.iterdir()):
                if not entry.is_dir():
                    continue
                index_path = entry / "index.jsonl"
                bases.append(
                    {
                        "id": entry.name,
                        "chunk_count": count_knowledge_chunks(
                            settings.data_dir,
                            user_id=user_id,
                            knowledge_base_id=entry.name,
                        ),
                        "has_index": index_path.exists(),
                    }
                )
        return bases

    return {"bases": await asyncio.to_thread(_scan)}


@router.post("/bases")
async def create_base(
    body: CreateBaseRequest,
    user_id: Annotated[str, Depends(get_current_user_id)],
    settings: Annotated[ForgeSettings, Depends(get_settings)],
) -> dict:
    kb_id = validate_kb_id(body.knowledge_base_id)
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
    file: Annotated[UploadFile, File(...)],
) -> dict:
    try:
        uploads = safe_join(settings.data_dir, "uploads", user_id)
        uploads.mkdir(parents=True, exist_ok=True)
    except SecurityError as exc:
        raise HTTPException(400, str(exc)) from exc

    raw_name = Path(file.filename or "upload.txt").name
    if not _FILENAME_RE.match(raw_name):
        raise HTTPException(
            400,
            "Filename must contain only letters, numbers, dots, hyphens, or underscores",
        )

    content = await file.read()
    if len(content) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            400, f"File exceeds {_MAX_UPLOAD_BYTES // (1024 * 1024)} MiB limit"
        )

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
    validate_kb_id(body.knowledge_base_id)
    try:
        uploads = safe_join(settings.data_dir, "uploads", user_id)
        uploads.mkdir(parents=True, exist_ok=True)
    except SecurityError as exc:
        raise HTTPException(400, str(exc)) from exc

    # Ephemeral orchestrator job (F4-06b): ingest is awaited in-request;
    # durable state is the KB files under the user sandbox, not a job table.
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
    settings: Annotated[ForgeSettings, Depends(get_settings)],
) -> dict:
    """Fast-path keyword retrieve (no job orchestration overhead)."""
    from forge.services.knowledge_context import retrieve_knowledge_chunks
    from forge.tools.sanitize import wrap_tool_result

    kb_id = validate_kb_id(body.knowledge_base_id)
    chunks = await asyncio.to_thread(
        retrieve_knowledge_chunks,
        settings.data_dir,
        user_id=user_id,
        knowledge_base_id=kb_id,
        query=body.query,
        top_k=body.top_k,
    )
    results = [
        {**c, "text": wrap_tool_result(f"kb:{kb_id}", c["text"])} for c in chunks
    ]
    return {"results": results}
