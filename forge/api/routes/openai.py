"""OpenAI-compatible API for local inference."""

from __future__ import annotations

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
from forge.security.autodefense import defense_enabled, scan_output
from forge.security.openai_auth import get_openai_user_id
from forge.services.llm_output import StreamingOutputSanitizer, chunk_sanitized_output, sanitize_llm_output
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


_UNTRUSTED_OPENAI_ROLES = frozenset({"tool", "function", "system"})


def _resolve_payload(body: ChatCompletionRequest, model_path: str | None) -> dict[str, Any]:
    messages = []
    for m in body.messages:
        role = m.role.lower()
        if role in _UNTRUSTED_OPENAI_ROLES:
            raise HTTPException(400, f"Untrusted message role: {m.role}")
        content = m.content if isinstance(m.content, str) else json.dumps(m.content)
        messages.append({"role": m.role, "content": content})
    if not any(m["role"].lower() == "user" for m in messages):
        raise HTTPException(400, "At least one user message is required")
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
    if body.model in ("default", "seiso"):
        models = await db.list_models(user_id)
        for row in models:
            path = await resolve_model_path(
                db, user_id, model_id=row["id"], model_path=None, data_dir=settings.data_dir
            )
            if path:
                return path
        raise HTTPException(400, "No local model available — download from Hub")

    path = await resolve_model_path(
        db, user_id, model_id=body.model, model_path=None, data_dir=settings.data_dir
    )
    if path:
        return path
    raise HTTPException(400, "No local model available — download from Hub")


@router.get("/v1/models")
async def list_models(
    user_id: Annotated[str, Depends(get_openai_user_id)],
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
    user_id: Annotated[str, Depends(get_openai_user_id)],
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
        use_defense = defense_enabled(settings)

        async def sse_stream():
            parts: list[str] = []
            try:
                stream_guard = StreamingOutputSanitizer()
                async for token in orchestrator.stream_local(payload):
                    parts.append(token)
                    if not use_defense:
                        for safe_token in stream_guard.feed(token):
                            chunk = {
                                "id": completion_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": body.model,
                                "choices": [
                                    {"index": 0, "delta": {"content": safe_token}, "finish_reason": None}
                                ],
                            }
                            yield f"data: {json.dumps(chunk)}\n\n"
                content = sanitize_llm_output("".join(parts))
                if use_defense and parts:
                    content, _ = await scan_output(
                        payload.get("messages", []),
                        content,
                        session_id=payload.get("thread_id"),
                        user_id=user_id,
                        settings=settings,
                    )
                output_tokens = chunk_sanitized_output(content) if use_defense else stream_guard.finish()
                for token in output_tokens:
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
                content = sanitize_llm_output(content)
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

    content = sanitize_llm_output(job.result.get("content", ""))
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
