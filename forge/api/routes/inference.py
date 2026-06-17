"""Chat and inference routes with SSE streaming."""

from __future__ import annotations

import json
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from forge.api.deps import get_db, get_inference_orchestrator
from forge.config import ForgeSettings, get_settings
from forge.db.store import Database
from forge.orchestrators.inference import InferenceOrchestrator
from forge.security.auth import get_current_user_id
from forge.security.autodefense import DefenseBlockedError, defense_enabled, scan_output
from forge.services.inference_models import list_inference_options, resolve_chat_target
from forge.services.mcp_access import validate_mcp_server_ids
from forge.services.models import resolve_model_path
from seiso.inference.backends import BACKEND_OLLAMA

router = APIRouter(prefix="/inference", tags=["inference"])


class ChatRequest(BaseModel):
    thread_id: str | None = None
    model_id: str | None = None
    model_path: str | None = None
    ollama_model: str | None = None
    inference_backend: str = Field(default="auto", description="auto | llamacpp | ollama | mlx | torch")
    messages: list[dict[str, str]] = Field(default_factory=list)
    max_tokens: int = Field(default=512, ge=1, le=8192)
    stream: bool = True
    tools: bool = False
    allow_code_exec: bool = False
    mcp_server_ids: list[str] = Field(default_factory=list)
    provider_id: str | None = None
    defense: bool | None = Field(
        default=None,
        description="Enable AutoDefense scan (requires SEISO_AUTODEFENSE_ENABLED). null=on when server enabled.",
    )


class ThreadCreate(BaseModel):
    title: str = "New chat"
    model_id: str | None = None


@router.post("/threads")
async def create_thread(
    body: ThreadCreate,
    user_id: Annotated[str, Depends(get_current_user_id)],
    db: Annotated[Database, Depends(get_db)],
) -> dict:
    return await db.create_thread(user_id, body.title, body.model_id)


@router.get("/threads")
async def list_threads(
    user_id: Annotated[str, Depends(get_current_user_id)],
    db: Annotated[Database, Depends(get_db)],
) -> list[dict]:
    return await db.list_threads(user_id)


@router.get("/models")
async def inference_models(
    user_id: Annotated[str, Depends(get_current_user_id)],
    db: Annotated[Database, Depends(get_db)],
    settings: Annotated[ForgeSettings, Depends(get_settings)],
) -> dict[str, Any]:
    """Unified model dropdown: HF Hub inventory, CLI paths, fine-tune/export outputs, Ollama."""
    options = await list_inference_options(db, user_id, ollama_base_url=settings.ollama_base_url)
    return {"models": options, "total": len(options)}


@router.get("/threads/{thread_id}/messages")
async def get_thread_messages(
    thread_id: str,
    user_id: Annotated[str, Depends(get_current_user_id)],
    db: Annotated[Database, Depends(get_db)],
) -> list[dict]:
    if not await db.get_thread_for_user(thread_id, user_id):
        raise HTTPException(404, "Thread not found")
    return await db.get_messages(thread_id)


@router.post("/chat")
async def chat(
    body: ChatRequest,
    user_id: Annotated[str, Depends(get_current_user_id)],
    db: Annotated[Database, Depends(get_db)],
    orchestrator: Annotated[InferenceOrchestrator, Depends(get_inference_orchestrator)],
    settings: Annotated[ForgeSettings, Depends(get_settings)],
):
    if body.thread_id and not await db.get_thread_for_user(body.thread_id, user_id):
        raise HTTPException(404, "Thread not found")

    if body.tools and not settings.allow_tools:
        raise HTTPException(403, "Tools are disabled on this server")

    if body.allow_code_exec and not settings.allow_code_exec:
        raise HTTPException(403, "Code execution is disabled on this server")

    if body.defense is True and not settings.autodefense_enabled:
        raise HTTPException(400, "AutoDefense is not enabled on this server (SEISO_AUTODEFENSE_ENABLED)")

    job_id = orchestrator.create_job(user_id=user_id)
    payload = body.model_dump()
    payload["user_id"] = user_id

    if body.tools and body.mcp_server_ids:
        await validate_mcp_server_ids(db, orchestrator.mcp, user_id, body.mcp_server_ids)

    if body.provider_id:
        prov = await db.get_provider(body.provider_id, user_id)
        if not prov:
            raise HTTPException(404, "Provider not found")
        payload["provider"] = {
            "provider_type": prov["provider_type"],
            "config": json.loads(prov["config_json"]),
        }
    if not body.provider_id:
        if body.model_path and body.model_id:
            raise HTTPException(403, "Provide model_id or model_path, not both")
        if body.model_path:
            try:
                path = await resolve_model_path(
                    db,
                    user_id,
                    model_id=None,
                    model_path=body.model_path,
                    data_dir=settings.data_dir,
                )
            except HTTPException:
                raise
            if not path:
                raise HTTPException(400, "Invalid model_path")
            payload["model_path"] = path
            payload["inference_backend"] = body.inference_backend
            payload["ollama_base_url"] = settings.ollama_base_url
        elif body.model_id:
            options = await list_inference_options(db, user_id, ollama_base_url=settings.ollama_base_url)
            selected = next((o for o in options if o["id"] == body.model_id), None)
            try:
                target = resolve_chat_target(
                    selected,
                    model_id=body.model_id,
                    ollama_model=body.ollama_model,
                    inference_backend=body.inference_backend,
                )
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc

            payload["inference_backend"] = target.get("inference_backend", body.inference_backend)
            payload["ollama_model"] = target.get("ollama_model")
            payload["model_format"] = target.get("model_format")
            payload["ollama_base_url"] = settings.ollama_base_url

            if target.get("inference_backend") == BACKEND_OLLAMA and not target.get("model_path"):
                if not payload.get("ollama_model"):
                    raise HTTPException(400, "Select an Ollama model")
            else:
                path = target.get("model_path")
                if not path and body.model_id:
                    path = await resolve_model_path(
                        db,
                        user_id,
                        model_id=body.model_id,
                        model_path=body.model_path,
                        data_dir=settings.data_dir,
                    )
                if not path:
                    raise HTTPException(400, "Select a model from inventory or provide model_path")
                payload["model_path"] = path
        else:
            raise HTTPException(400, "Select a model from inventory or provide model_path")

    if body.thread_id and body.messages:
        last = body.messages[-1]
        if last.get("role") == "user":
            await db.add_message(body.thread_id, "user", last["content"])

    if body.stream:
        can_stream_local = not body.tools and not body.provider_id
        use_defense = defense_enabled(settings, request_flag=body.defense)

        async def event_gen():
            if can_stream_local:
                parts: list[str] = []
                backend_label = payload.get("inference_backend") or "local"
                try:
                    orchestrator._emit_log(job_id, f"Streaming inference ({backend_label})")
                    async for token in orchestrator.stream_local(payload):
                        parts.append(token)
                        yield {"event": "token", "data": token}
                    content = "".join(parts)
                    if use_defense and content:
                        content, defense_result = await scan_output(
                            payload.get("messages", []),
                            content,
                            session_id=body.thread_id or job_id,
                            user_id=user_id,
                            settings=settings,
                        )
                        if not defense_result.unavailable:
                            yield {"event": "defense", "data": json.dumps(defense_result.to_dict())}
                    if body.thread_id:
                        await db.add_message(body.thread_id, "assistant", content)
                    yield {"event": "message", "data": content}
                    yield {"event": "done", "data": job_id}
                except DefenseBlockedError as exc:
                    yield {"event": "error", "data": str(exc)}
                    if exc.result:
                        yield {"event": "defense", "data": json.dumps(exc.result.to_dict())}
                except Exception as exc:
                    yield {"event": "error", "data": str(exc)}
                return

            await orchestrator.start(job_id, payload)
            async for line in orchestrator.stream_logs(job_id):
                yield {"event": "log", "data": line}
            job = await orchestrator.wait_for(job_id)
            if job and job.status.value == "failed":
                yield {"event": "error", "data": job.error or "Inference failed"}
                if job.result.get("defense"):
                    yield {"event": "defense", "data": json.dumps(job.result["defense"])}
            elif job and job.result.get("content"):
                content = job.result["content"]
                if body.thread_id:
                    await db.add_message(body.thread_id, "assistant", content)
                yield {"event": "message", "data": content}
                if job.result.get("defense"):
                    yield {"event": "defense", "data": json.dumps(job.result["defense"])}
            yield {"event": "done", "data": job_id}

        return EventSourceResponse(event_gen())

    await orchestrator.start(job_id, payload)
    job = await orchestrator.wait_for(job_id)
    if not job:
        raise HTTPException(500, "Job lost")
    if job.status.value == "failed":
        detail = job.error or "Inference failed"
        if job.result.get("defense"):
            raise HTTPException(403, detail, headers={"X-Seiso-Defense": json.dumps(job.result["defense"])})
        raise HTTPException(500, detail)
    if body.thread_id and job.result.get("content"):
        await db.add_message(body.thread_id, "assistant", job.result["content"])
    return {"job_id": job_id, **job.result}
