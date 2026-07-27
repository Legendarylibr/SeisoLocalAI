"""Compat API for local + multi-GPU provider inference (chat-completions wire protocol).

External agents (Cursor, Continue, custom clients) use:

  Base URL: http://127.0.0.1:8765/v1
  Auth:     Bearer <SEISO_DATA_DIR/.inference_api_key>

Local inventory models keep working. Multi-GPU (managed local vLLM or cloud)
appears as additional ``/v1/models`` entries (``provider:<id>`` and optional
upstream model alias) and routes through the same completions endpoint.
"""

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
from forge.api.schemas.compat import ChatCompletionRequest
from forge.config import ForgeSettings, get_settings
from forge.db.store import Database
from forge.orchestrators.inference import InferenceOrchestrator
from forge.security.audit import audit_event, hash_audit_payload
from forge.security.compat_auth import CompatIdentity, get_compat_identity, get_compat_user_id
from forge.services.compat_chat import (
    estimate_token_count,
    prepare_compat_chat_payload,
    prompt_token_estimate,
)
from forge.services.compat_providers import list_compat_provider_models
from forge.services.llm_output import StreamingOutputSanitizer, sanitize_llm_output

_prepare_compat_chat_payload = prepare_compat_chat_payload

router = APIRouter(tags=["compat"])
logger = logging.getLogger(__name__)


@router.get("/v1/models")
async def list_models(
    user_id: Annotated[str, Depends(get_compat_user_id)],
    db: Annotated[Database, Depends(get_db)],
) -> dict:
    models = await db.list_models(user_id)
    created = int(time.time())
    data = [
        {
            "id": m["id"],
            "object": "model",
            "created": created,
            "owned_by": "seiso",
        }
        for m in models
    ]
    # Additive: multi-GPU / provider targets for external agents.
    provider_models = await list_compat_provider_models(db, user_id)
    seen = {d["id"] for d in data}
    for entry in provider_models:
        if entry["id"] not in seen:
            data.append(entry)
            seen.add(entry["id"])
    return {
        "object": "list",
        "data": data,
    }


@router.post("/v1/chat/completions")
async def chat_completions(
    body: ChatCompletionRequest,
    identity: Annotated[CompatIdentity, Depends(get_compat_identity)],
    db: Annotated[Database, Depends(get_db)],
    orchestrator: Annotated[InferenceOrchestrator, Depends(get_inference_orchestrator)],
    settings: Annotated[ForgeSettings, Depends(get_settings)],
):
    """Compat chat endpoint for Cursor, Continue, and other clients."""
    user_id = identity.user_id
    if body.tools:
        if not settings.allow_compat_tools:
            raise HTTPException(403, "Tool calling is disabled on the Compat API")
        if not identity.tools_allowed:
            raise HTTPException(
                403,
                "Inference API key is chat-only; use a session JWT for Compat tools",
            )

    payload = await prepare_compat_chat_payload(body, user_id, db, settings)
    payload["user_id"] = user_id
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    use_provider = bool(payload.get("provider"))
    audit_event(
        "compat_chat",
        user_id=user_id,
        auth_method=identity.auth_method,
        tools=bool(body.tools),
        stream=bool(body.stream),
        model=body.model,
        message_count=len(body.messages),
        messages_sha256=hash_audit_payload(
            [
                {
                    "role": m.role,
                    "content": m.content
                    if isinstance(m.content, str)
                    else str(m.content),
                }
                for m in body.messages
            ]
        ),
    )

    # Local loads need free GPU; multi-GPU provider (local managed or cloud) does not
    # use the in-process model pool.
    if not use_provider:
        _assert_inference_gpu_available()

    use_local_stream = body.stream and not body.tools and not use_provider
    use_provider_stream = body.stream and not body.tools and use_provider

    if use_local_stream:
        gen_epoch = _begin_generation_or_raise(orchestrator, user_id)

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
                logger.exception("Compat API inference stream failed")
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
                orchestrator.end_generation_for_user(user_id, epoch=gen_epoch)

        return StreamingResponse(sse_stream(), media_type="text/event-stream")

    if use_provider_stream:
        gen_epoch = _begin_generation_or_raise(orchestrator, user_id)

        async def provider_sse_stream():
            sanitizer = StreamingOutputSanitizer(strip_tool_calls=not body.tools)
            try:
                async for token in orchestrator.stream_provider(payload):
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
                logger.exception("Compat API provider stream failed")
                await orchestrator.cancel_generation_for_user(user_id)
                err = {
                    "error": {
                        "message": str(exc) or "Provider stream failed",
                        "type": "server_error",
                    }
                }
                yield f"data: {json.dumps(err)}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                orchestrator.end_generation_for_user(user_id, epoch=gen_epoch)

        return StreamingResponse(provider_sse_stream(), media_type="text/event-stream")

    job_id = orchestrator.create_job(user_id=user_id)
    gen_epoch = _begin_generation_or_raise(orchestrator, user_id)
    try:
        await orchestrator.start(job_id, payload)
    except asyncio.CancelledError:
        await orchestrator.cancel_generation_for_user(user_id)
        raise
    except Exception:
        await orchestrator.cancel_generation_for_user(user_id)
        raise

    if body.stream:

        async def job_sse_stream():
            # Tools/job path: OpenAI clients expect chat.completion.chunk only —
            # drain logs silently, then emit a single content delta.
            try:
                async for _line in orchestrator.stream_logs(job_id):
                    pass
                job = await orchestrator.wait_for(job_id)
                content = job.result.get("content", "") if job and job.result else ""
                if job and job.status.value == "failed":
                    err = {
                        "error": {
                            "message": job.error or "Inference failed",
                            "type": "server_error",
                        }
                    }
                    yield f"data: {json.dumps(err)}\n\n"
                elif job and job.status.value == "cancelled":
                    err = {
                        "error": {
                            "message": "Inference cancelled",
                            "type": "cancelled",
                        }
                    }
                    yield f"data: {json.dumps(err)}\n\n"
                elif content:
                    content = sanitize_llm_output(content, strip_tool_calls=not body.tools)
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
                orchestrator.end_generation_for_user(user_id, epoch=gen_epoch)

        return StreamingResponse(job_sse_stream(), media_type="text/event-stream")

    try:
        job = await orchestrator.wait_for(job_id)
    except asyncio.CancelledError:
        await orchestrator.cancel_generation_for_user(user_id)
        raise
    except Exception:
        await orchestrator.cancel_generation_for_user(user_id)
        raise
    finally:
        orchestrator.end_generation_for_user(user_id, epoch=gen_epoch)
    if not job or job.status.value == "failed":
        raise HTTPException(500, job.error if job else "Inference failed")
    if job.status.value == "cancelled":
        raise HTTPException(409, "Inference cancelled")

    content = sanitize_llm_output(job.result.get("content", ""), strip_tool_calls=not body.tools)
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
