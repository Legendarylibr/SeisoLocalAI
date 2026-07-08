"""OpenAI-compatible API for local inference."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from forge.api.deps import get_db, get_inference_orchestrator
from forge.api.routes._inference_common import (
    _assert_inference_gpu_available,
    _begin_generation_or_raise,
)
from forge.api.schemas.openai import ChatCompletionRequest
from forge.config import ForgeSettings, get_settings
from forge.db.store import Database
from forge.orchestrators.inference import InferenceOrchestrator
from forge.security.openai_auth import get_openai_user_id
from forge.services.llm_output import StreamingOutputSanitizer, sanitize_llm_output
from forge.services.openai_chat import (
    estimate_token_count,
    prepare_openai_chat_payload,
    prompt_token_estimate,
)

_prepare_openai_chat_payload = prepare_openai_chat_payload

router = APIRouter(tags=["openai"])
logger = logging.getLogger(__name__)


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

    payload = await prepare_openai_chat_payload(body, user_id, db, settings)
    payload["user_id"] = user_id
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    use_local_stream = body.stream and not body.tools

    if use_local_stream:
        _begin_generation_or_raise(orchestrator, user_id)

        async def sse_stream():
            sanitizer = StreamingOutputSanitizer(strip_tool_calls=not body.tools)
            raw_parts: list[str] = []
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
            finally:
                orchestrator.end_generation_for_user(user_id)

        return StreamingResponse(job_sse_stream(), media_type="text/event-stream")

    try:
        job = await orchestrator.wait_for(job_id)
    except Exception:
        await orchestrator.cancel_generation_for_user(user_id)
        raise
    if not job or job.status.value == "failed":
        raise HTTPException(500, job.error if job else "Inference failed")

    content = sanitize_llm_output(job.result.get("content", ""), strip_tool_calls=bool(body.tools))
    prompt_tokens = prompt_token_estimate(body.messages)
    completion_tokens = estimate_token_count(content)
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
