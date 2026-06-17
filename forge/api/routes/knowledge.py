"""Knowledge base / RAG routes."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from forge.api.deps import get_knowledge_orchestrator
from forge.config import ForgeSettings, get_settings
from forge.orchestrators.knowledge import KnowledgeOrchestrator
from forge.security.auth import get_current_user_id
from seiso.security import SecurityError, safe_join

router = APIRouter(prefix="/knowledge", tags=["knowledge"])


class IngestRequest(BaseModel):
    knowledge_base_id: str
    source_path: str


class RetrieveRequest(BaseModel):
    knowledge_base_id: str
    query: str
    top_k: int = Field(default=5, ge=1, le=20)


@router.post("/ingest")
async def ingest(
    body: IngestRequest,
    user_id: Annotated[str, Depends(get_current_user_id)],
    orchestrator: Annotated[KnowledgeOrchestrator, Depends(get_knowledge_orchestrator)],
    settings: Annotated[ForgeSettings, Depends(get_settings)],
) -> dict:
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
    job_id = orchestrator.create_job(user_id=user_id)
    payload = {"action": "retrieve", "user_id": user_id, **body.model_dump()}
    await orchestrator.start(job_id, payload)
    job = await orchestrator.wait_for(job_id)
    return job.result if job else {"results": []}
