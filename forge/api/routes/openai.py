"""OpenAI-compatible API for local inference."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from forge.api.deps import get_db, get_inference_orchestrator
from forge.api.routes._inference_common import (
    _assert_inference_gpu_available,
    _begin_generation_or_raise,
)
from forge.config import ForgeSettings, get_settings
from forge.db.store import Database
from forge.orchestrators.inference import InferenceOrchestrator
from forge.security.openai_auth import get_openai_user_id
from forge.services.inference_chat import prepare_local_chat_target
from forge.services.llm_output import StreamingOutputSanitizer, sanitize_llm_output
from forge.services.user_paths import is_local_filesystem_path
from forge.tools.sanitize import normalize_text

router = APIRouter(tags=["openai"])
logger = logging.getLogger(__name__)


class ChatMessage(BaseModel):
    role: str
    content: str | list[Any] = ""


class ChatCompletionRequest(BaseModel):
    model: str = Field(default="default")
    messages: list[ChatMessage] = Field(default_factory=list)
    max_tokens: int | None = Field(default=2048, ge=1, le=8192)
    temperature: float = Field(default=0.7, ge=0, le=2)
    stream: bool = False
    tools: list[dict] | None = None


_UNTRUSTED_OPENAI_ROLES = frozenset({"tool", "function", "system", "developer"})
_UNVERIFIED_ASSISTANT_PREFIX = "[UNVERIFIED_PRIOR_ASSISTANT]\n"


def _normalize_openai_messages(body: ChatCompletionRequest) -> list[dict[str, str]]:
    """Reject privileged roles; downgrade client assistant turns to unverified user data."""
    if not body.messages:
        raise HTTPException(400, "At least one user message is required")
    if body.messages[-1].role.lower() != "user":
        raise HTTPException(400, "Last message must be from user")

    messages: list[dict[str, str]] = []
    for m in body.messages:
        role = m.role.lower()
        if role in _UNTRUSTED_OPENAI_ROLES:
            raise HTTPException(400, f"Untrusted message role: {m.role}")
        content = normalize_text(m.content if isinstance(m.content, str) else json.dumps(m.content))
        if role == "assistant":
            messages.append(
                {
                    "role": "user",
                    "content": f"{_UNVERIFIED_ASSISTANT_PREFIX}{content}",
                }
            )
            continue
        if role != "user":
            raise HTTPException(400, f"Unsupported message role: {m.role}")
        messages.append({"role": "user", "content": content})
    if not messages:
        raise HTTPException(400, "At least one user message is required")
    if messages[-1]["role"] != "user":
        raise HTTPException(400, "Last message must be from user")
    return messages


def _estimate_token_count(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0
    return max(1, len(stripped.split()))


def _prompt_token_estimate(messages: list[ChatMessage]) -> int:
    return sum(
        _estimate_token_count(m.content if isinstance(m.content, str) else json.dumps(m.content))
        for m in messages
    )


async def _prepare_openai_chat_payload(
    body: ChatCompletionRequest,
    user_id: str,
    db: Database,
    settings: ForgeSettings,
) -> dict[str, Any]:
    """Resolve and sanitize via the shared local-chat path."""
    messages = _normalize_openai_messages(body)
    max_tokens = body.max_tokens or 512

    if body.model in ("default", "seiso"):
        from forge.services.inference_models import list_inference_options

        options = await list_inference_options(db, user_id, hardware_aware=False)
        selected = next(
            (
                o
                for o in options
                if o.get("selectable", True)
                and (o.get("format") or "").lower() == "gguf"
                and o.get("kind") == "local"
            ),
            None,
        )
        if selected is None:
            selected = next(
                (o for o in options if o.get("selectable", True) and o.get("kind") == "local"),
                None,
            )
        if selected is None:
            raise HTTPException(400, "No local model available — download from Hub")
        target = await prepare_local_chat_target(
            db,
            user_id,
            settings,
            model_id=selected["id"],
            inference_backend="auto",
            max_tokens=max_tokens,
            messages=messages,
            check_memory=True,
            sanitize=True,
        )
    elif is_local_filesystem_path(body.model):
        target = await prepare_local_chat_target(
            db,
            user_id,
            settings,
            model_path=body.model,
            inference_backend="auto",
            max_tokens=max_tokens,
            messages=messages,
            check_memory=True,
            sanitize=True,
        )
    else:
        match = await db.get_model(body.model, user_id)
        if match is None:
            match = await db.get_model_by_name(user_id, body.model)
        if not match:
            raise HTTPException(404, f"Model not found in inventory: {body.model}")
        target = await prepare_local_chat_target(
            db,
            user_id,
            settings,
            model_id=match["id"],
            inference_backend="auto",
            max_tokens=max_tokens,
            messages=messages,
            check_memory=True,
            sanitize=True,
        )

    payload: dict[str, Any] = {
        "model_path": target.get("model_path"),
        "messages": messages,
        "max_tokens": target.get("max_tokens", max_tokens),
        "temperature": body.temperature,
        "tools": bool(body.tools),
        "inference_backend": target.get("inference_backend", "auto"),
    }
    if target.get("model_format"):
        payload["model_format"] = target["model_format"]
    if target.get("model_metadata"):
        payload["model_metadata"] = target["model_metadata"]
    if target.get("n_ctx") is not None:
        payload["n_ctx"] = target["n_ctx"]
    return payload


@router.get("/v1/models")
async def list_models(
    user_id: Annotated[str, Depends(get_openai_user_id)],
    db: Annotated[Database, Depends(get_db)],
) -> dict:
    models = await db.list_models(user_id)
    created = int(time.time())
    return {
        "object": "list",
        "data": [
            {
                "id": m["id"],
                "object": "model",
                "created": created,
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

    _assert_inference_gpu_available()

    payload = await _prepare_openai_chat_payload(body, user_id, db, settings)
    payload["user_id"] = user_id
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    use_local_stream = body.stream and not body.tools

    if use_local_stream:
        _begin_generation_or_raise(orchestrator, user_id)

        async def sse_stream():
            sanitizer = StreamingOutputSanitizer(strip_tool_calls=not body.tools)
            raw_parts: list[str] = []
            completed = False
            try:
                async for token in orchestrator.stream_local(payload):
                    raw_parts.append(token)
                    for chunk in sanitizer.feed(token):
                        chunk_payload = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": body.model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": chunk},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(chunk_payload)}\n\n"
                for chunk in sanitizer.finish():
                    chunk_payload = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": body.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": chunk},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk_payload)}\n\n"
                final = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": body.model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                yield f"data: {json.dumps(final)}\n\n"
                yield "data: [DONE]\n\n"
                completed = True
            except asyncio.CancelledError:
                await orchestrator.cancel_generation_for_user(user_id)
                raise
            except Exception as exc:
                logger.exception("OpenAI-compatible inference stream failed")
                await orchestrator.cancel_generation_for_user(user_id)
                err = {
                    "error": {
                        "message": str(exc) or "Inference stream failed",
                        "type": "server_error",
                    }
                }
                yield f"data: {json.dumps(err)}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                if completed:
                    orchestrator.end_generation_for_user(user_id)

        return StreamingResponse(sse_stream(), media_type="text/event-stream")

    job_id = orchestrator.create_job(user_id=user_id)
    _begin_generation_or_raise(orchestrator, user_id)
    try:
        await orchestrator.start(job_id, payload)
    except Exception:
        await orchestrator.cancel_generation_for_user(user_id)
        raise

    if body.stream:

        async def job_sse_stream():
            try:
                async for line in orchestrator.stream_logs(job_id):
                    yield f"data: {json.dumps({'log': line})}\n\n"
                job = await orchestrator.wait_for(job_id)
                content = job.result.get("content", "") if job and job.result else ""
                if job and job.status.value == "failed":
                    yield f"data: {json.dumps({'error': job.error or 'Inference failed'})}\n\n"
                elif content:
                    content = sanitize_llm_output(content, strip_tool_calls=bool(body.tools))
                    chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": body.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": content},
                                "finish_reason": "stop",
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"
            except asyncio.CancelledError:
                await orchestrator.cancel_generation_for_user(user_id)
                raise
            except Exception as exc:
                await orchestrator.cancel_generation_for_user(user_id)
                yield f"data: {json.dumps({'error': str(exc) or 'Inference failed'})}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(job_sse_stream(), media_type="text/event-stream")

    try:
        job = await orchestrator.wait_for(job_id)
    except Exception:
        await orchestrator.cancel_generation_for_user(user_id)
        raise
    if not job or job.status.value == "failed":
        raise HTTPException(500, job.error if job else "Inference failed")

    content = sanitize_llm_output(job.result.get("content", ""), strip_tool_calls=bool(body.tools))
    prompt_tokens = _prompt_token_estimate(body.messages)
    completion_tokens = _estimate_token_count(content)
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
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
    )
