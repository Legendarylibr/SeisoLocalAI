"""OpenAI-compatible API for local inference."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from forge.api.deps import get_db, get_inference_orchestrator
from forge.config import ForgeSettings, get_settings
from forge.db.store import Database
from forge.orchestrators.inference import InferenceOrchestrator
from forge.security.auth import get_current_user_id
from forge.services.models import resolve_model_path

router = APIRouter(tags=["openai"])


class ChatMessage(BaseModel):
    role: str
    content: str | list[Any] = ""


class ChatCompletionRequest(BaseModel):
    model: str = Field(default="default")
    messages: list[ChatMessage] = Field(default_factory=list)
    max_tokens: int | None = Field(default=512, ge=1, le=8192)
    temperature: float = Field(default=0.7, ge=0, le=2)
    stream: bool = False
    tools: list[dict] | None = None


def _resolve_payload(body: ChatCompletionRequest, model_path: str | None) -> dict[str, Any]:
    messages = []
    for m in body.messages:
        content = m.content if isinstance(m.content, str) else json.dumps(m.content)
        messages.append({"role": m.role, "content": content})
    return {
        "model_path": model_path,
        "messages": messages,
        "max_tokens": body.max_tokens or 512,
        "temperature": body.temperature,
        "tools": bool(body.tools),
    }


async def _resolve_openai_model_path(
    body: ChatCompletionRequest,
    user_id: str,
    db: Database,
    settings: ForgeSettings,
) -> str:
    model_id = None if body.model in ("default", "seiso") else body.model
    try:
        path = await resolve_model_path(
            db, user_id, model_id=model_id, model_path=None, data_dir=settings.data_dir
        )
    except HTTPException:
        path = None
    if path:
        return path
    inv = await db.list_models(user_id)
    if inv:
        return inv[0]["path"]
    raise HTTPException(400, "No local model available — download from Hub")


@router.get("/v1/models")
async def list_models(
    user_id: Annotated[str, Depends(get_current_user_id)],
    db: Annotated[Database, Depends(get_db)],
) -> dict:
    models = await db.list_models(user_id)
    return {
        "object": "list",
        "data": [
            {
                "id": m["id"],
                "object": "model",
                "created": int(time.time()),
                "owned_by": "seiso",
            }
            for m in models
        ],
    }


@router.post("/v1/chat/completions")
async def chat_completions(
    body: ChatCompletionRequest,
    user_id: Annotated[str, Depends(get_current_user_id)],
    db: Annotated[Database, Depends(get_db)],
    orchestrator: Annotated[InferenceOrchestrator, Depends(get_inference_orchestrator)],
    settings: Annotated[ForgeSettings, Depends(get_settings)],
):
    """OpenAI-compatible chat endpoint for Cursor, Continue, and other clients."""
    if body.tools and not settings.allow_openai_tools:
        raise HTTPException(403, "Tool calling is disabled on the OpenAI-compatible API")

    path = await _resolve_openai_model_path(body, user_id, db, settings)
    payload = _resolve_payload(body, path)
    payload["user_id"] = user_id
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    use_local_stream = body.stream and not body.tools

    if use_local_stream:

        async def sse_stream():
            try:
                async for token in orchestrator.stream_local(payload):
                    chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": body.model,
                        "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                final = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": body.model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                yield f"data: {json.dumps(final)}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as exc:
                err = {"error": {"message": str(exc), "type": "server_error"}}
                yield f"data: {json.dumps(err)}\n\n"

        return StreamingResponse(sse_stream(), media_type="text/event-stream")

    job_id = orchestrator.create_job(user_id=user_id)
    await orchestrator.start(job_id, payload)

    if body.stream:

        async def job_sse_stream():
            async for line in orchestrator.stream_logs(job_id):
                yield f"data: {json.dumps({'log': line})}\n\n"
            job = await orchestrator.wait_for(job_id)
            content = job.result.get("content", "") if job and job.result else ""
            if job and job.status.value == "failed":
                yield f"data: {json.dumps({'error': job.error or 'Inference failed'})}\n\n"
            elif content:
                chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": body.model,
                    "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": "stop"}],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(job_sse_stream(), media_type="text/event-stream")

    job = await orchestrator.wait_for(job_id)
    if not job or job.status.value == "failed":
        raise HTTPException(500, job.error if job else "Inference failed")

    content = job.result.get("content", "")
    return JSONResponse(
        {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": body.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": len(content.split()),
                "total_tokens": len(content.split()),
            },
        }
    )
