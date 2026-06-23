"""Chat and inference routes with SSE streaming."""

from __future__ import annotations

import asyncio
import json
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from forge.api.deps import get_db, get_inference_orchestrator
from forge.config import ForgeSettings, get_settings
from forge.db.store import Database
from forge.orchestrators.inference import InferenceOrchestrator
from forge.security.auth import get_current_user_id
from forge.services.chat_messages import build_trusted_messages
from forge.services.download_progress import estimate_load_eta_seconds
from forge.services.hardware import hardware_profile
from forge.services.hf_cache_inventory import sync_hf_cache_inventory
from forge.services.inference_models import (
    get_inference_option,
    list_inference_options,
    resolve_chat_target,
)
from forge.services.knowledge_context import format_knowledge_context, retrieve_knowledge_chunks
from forge.services.llm_output import StreamingOutputSanitizer, sanitize_llm_output
from forge.services.model_router_client import ROUTER_MODEL_ID, fetch_router_status
from forge.services.models import resolve_model_path
from seiso.inference.backends import BACKEND_LLAMACPP, BACKEND_MLX, BACKEND_ROUTER, BACKEND_TORCH

router = APIRouter(prefix="/inference", tags=["inference"])

_POOL_BACKEND_BY_API_BACKEND = {
    BACKEND_LLAMACPP: "llama",
    "llama": "llama",
    BACKEND_MLX: "mlx",
    BACKEND_TORCH: "torch",
}


def _assert_inference_gpu_available() -> None:
    from forge.services.memory_release import assert_gpu_available_for_inference

    try:
        assert_gpu_available_for_inference()
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc


class ChatRequest(BaseModel):
    thread_id: str | None = None
    model_id: str | None = None
    model_path: str | None = None
    draft_model_id: str | None = None
    draft_model_path: str | None = None
    num_speculative_tokens: int | None = Field(default=None, ge=1, le=32)
    inference_backend: str = Field(
        default="auto", description="auto | llamacpp | mlx | torch"
    )
    messages: list[dict[str, str]] = Field(default_factory=list)
    max_tokens: int = Field(default=2048, ge=1, le=8192)
    n_ctx: int | None = Field(default=None, ge=2048, le=131072)
    temperature: float = Field(default=0.7, ge=0, le=2)
    top_p: float | None = Field(default=None, ge=0, le=1)
    stream: bool = True
    tools: bool = False
    allow_code_exec: bool = False
    provider_id: str | None = None
    knowledge_base_id: str | None = None
    router_model: str | None = Field(
        default=None,
        description="Optional explicit specialist model id for Smart Router",
    )


class ThreadCreate(BaseModel):
    title: str = "New chat"
    model_id: str | None = None


class PreloadRequest(BaseModel):
    model_id: str
    inference_backend: str = Field(
        default="auto", description="auto | llamacpp | mlx | torch"
    )


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
    """Unified model dropdown: HF Hub inventory, CLI paths, fine-tune/export outputs."""
    from forge.services.hardware import hardware_summary

    await sync_hf_cache_inventory(
        db,
        user_id,
        data_dir=settings.data_dir,
        hf_cache_dir=settings.hf_cache_dir,
    )
    profile = hardware_profile()
    options = await list_inference_options(
        db,
        user_id,
        profile=profile,
        model_router_enabled=settings.model_router_enabled,
    )
    return {
        "models": options,
        "total": len(options),
        "hardware_summary": hardware_summary(profile),
        "preferred_inference_backend": profile.get("preferred_inference_backend"),
        "local_only": True,
        "model_router": {
            "enabled": settings.model_router_enabled,
            "url": settings.model_router_url if settings.model_router_enabled else "",
            "model_id": ROUTER_MODEL_ID,
        },
    }



@router.get("/models/{model_id}/variants")
async def inference_model_variants(
    model_id: str,
    user_id: Annotated[str, Depends(get_current_user_id)],
    db: Annotated[Database, Depends(get_db)],
    settings: Annotated[ForgeSettings, Depends(get_settings)],
) -> dict[str, Any]:
    from forge.services.hf_auth import resolve_hf_token
    from forge.services.inference_variants import get_model_variants

    hf_token, _ = resolve_hf_token(
        user_id=user_id,
        data_dir=settings.data_dir,
        encryption_key=settings.hf_token_encryption_key,
        settings_token=settings.hf_token or None,
    )
    variants = await get_model_variants(db, user_id, model_id, hf_token=hf_token)
    if not await get_inference_option(db, user_id, model_id):
        raise HTTPException(404, "Model not found")
    return variants


@router.get("/router/status")
async def router_status(
    settings: Annotated[ForgeSettings, Depends(get_settings)],
) -> dict[str, Any]:
    if not settings.model_router_enabled:
        return {"enabled": False}
    try:
        status = await fetch_router_status(settings)
        return {"enabled": True, **status}
    except Exception as exc:
        raise HTTPException(502, f"Model router unavailable: {exc}") from exc


@router.delete("/threads/{thread_id}")
async def delete_thread(
    thread_id: str,
    user_id: Annotated[str, Depends(get_current_user_id)],
    db: Annotated[Database, Depends(get_db)],
) -> dict[str, str]:
    if not await db.delete_thread(thread_id, user_id):
        raise HTTPException(404, "Thread not found")
    return {"status": "deleted"}


@router.get("/session")
async def chat_session_info(
    user_id: Annotated[str, Depends(get_current_user_id)],
    db: Annotated[Database, Depends(get_db)],
) -> dict:
    thread_count = await db.count_threads(user_id)
    return {
        "thread_count": thread_count,
        "memory_encrypted": True,
        "clears_on_logout": True,
        "local_only": True,
    }


@router.get("/threads/{thread_id}/messages")
async def get_thread_messages(
    thread_id: str,
    user_id: Annotated[str, Depends(get_current_user_id)],
    db: Annotated[Database, Depends(get_db)],
) -> list[dict]:
    if not await db.get_thread_for_user(thread_id, user_id):
        raise HTTPException(404, "Thread not found")
    return await db.get_messages(thread_id)


@router.get("/context")
async def get_chat_context(
    user_id: Annotated[str, Depends(get_current_user_id)],
    db: Annotated[Database, Depends(get_db)],
    settings: Annotated[ForgeSettings, Depends(get_settings)],
    thread_id: str | None = None,
    max_tokens: int = Query(default=2048, ge=1, le=8192),
    n_ctx: int | None = Query(default=None, ge=2048, le=131072),
    tools: bool = False,
    knowledge_base_id: str | None = None,
    model_id: str | None = None,
) -> dict[str, Any]:
    """Context window usage for the chat UI."""
    from forge.api.routes.knowledge import _validate_kb_id
    from forge.services.chat_context import context_status_for_history
    from forge.services.inference_models import get_inference_option
    from forge.services.knowledge_context import format_knowledge_context, retrieve_knowledge_chunks

    history: list[dict] = []
    if thread_id:
        if not await db.get_thread_for_user(thread_id, user_id):
            raise HTTPException(404, "Thread not found")
        history = await db.get_messages(thread_id)

    model_path: str | None = None
    model_format: str | None = None
    model_name: str | None = None
    if model_id:
        selected = await get_inference_option(db, user_id, model_id)
        if selected:
            model_path = selected.get("path")
            model_format = selected.get("format")
            model_name = selected.get("name")

    knowledge_context: str | None = None
    if knowledge_base_id:
        kb_id = _validate_kb_id(knowledge_base_id)
        last_user = ""
        for msg in reversed(history):
            if msg.get("role") == "user":
                last_user = str(msg.get("content", "")).strip()
                break
        chunks = retrieve_knowledge_chunks(
            settings.data_dir,
            user_id=user_id,
            knowledge_base_id=kb_id,
            query=last_user,
        )
        knowledge_context = format_knowledge_context(chunks) or None

    return context_status_for_history(
        history,
        max_tokens=max_tokens,
        n_ctx=n_ctx,
        model_id=model_id,
        model_path=model_path,
        model_format=model_format,
        model_name=model_name,
        tools_enabled=tools,
        knowledge_context=knowledge_context,
    )


@router.post("/cancel")
async def cancel_inference(
    user_id: Annotated[str, Depends(get_current_user_id)],
    orchestrator: Annotated[InferenceOrchestrator, Depends(get_inference_orchestrator)],
) -> dict:
    """Abort in-flight generation and unload the active local model from VRAM."""
    try:
        return await orchestrator.cancel_and_unload_for_user(user_id)
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc


@router.post("/cancel-generation")
async def cancel_generation(
    user_id: Annotated[str, Depends(get_current_user_id)],
    orchestrator: Annotated[InferenceOrchestrator, Depends(get_inference_orchestrator)],
) -> dict:
    """Abort in-flight generation but keep the active local model warmed."""
    try:
        return await orchestrator.cancel_generation_for_user(user_id)
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc


@router.post("/preload")
async def preload_model(
    body: PreloadRequest,
    user_id: Annotated[str, Depends(get_current_user_id)],
    db: Annotated[Database, Depends(get_db)],
    orchestrator: Annotated[InferenceOrchestrator, Depends(get_inference_orchestrator)],
    settings: Annotated[ForgeSettings, Depends(get_settings)],
) -> dict[str, Any]:
    """Load a selected inventory model into the local inference engine."""
    _assert_inference_gpu_available()
    ctx = await _resolve_preload_context(
        db, user_id, settings, body.model_id, body.inference_backend
    )

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(
        None, lambda: _warm_local_model(orchestrator._runner, ctx["payload"])
    )
    status = orchestrator._runner._pool.status()
    return {"status": "loaded", "backend": ctx["backend"], **status}


@router.post("/preload/stream")
async def preload_model_stream(
    body: PreloadRequest,
    user_id: Annotated[str, Depends(get_current_user_id)],
    db: Annotated[Database, Depends(get_db)],
    orchestrator: Annotated[InferenceOrchestrator, Depends(get_inference_orchestrator)],
    settings: Annotated[ForgeSettings, Depends(get_settings)],
):
    _assert_inference_gpu_available()
    ctx = await _resolve_preload_context(
        db, user_id, settings, body.model_id, body.inference_backend
    )

    runner = orchestrator._runner
    pool = runner._pool
    target_path = ctx["payload"]["model_path"]
    size_bytes = ctx.get("size_bytes", 0)
    eta = estimate_load_eta_seconds(size_bytes)
    loop = asyncio.get_running_loop()

    async def event_gen():

        switching = orchestrator._runner._pool.would_switch_model(
            target_path, ctx["backend"]
        )
        if switching:
            yield {
                "event": "progress",
                "data": json.dumps(
                    {
                        "phase": "unloading",
                        "label": "Releasing previous model from VRAM",
                        "percent": 5,
                        "eta_seconds": 2,
                    }
                ),
            }
            await loop.run_in_executor(
                None,
                lambda: orchestrator._runner._pool.prepare_for_load(
                    target_path, ctx["backend"]
                ),
            )

        yield {
            "event": "progress",
            "data": json.dumps(
                {
                    "phase": "loading",
                    "label": f"Loading {ctx['model_name']} into inference engine",
                    "percent": 15,
                    "eta_seconds": eta,
                    "model_id": body.model_id,
                    "model_name": ctx["model_name"],
                    "backend": ctx["backend"],
                    "size_bytes": size_bytes,
                }
            ),
        }
        try:
            await loop.run_in_executor(None, lambda: _warm_local_model(runner, ctx["payload"]))
        except Exception as exc:
            yield {"event": "error", "data": str(exc)}
            return

        status = pool.status()
        yield {
            "event": "progress",
            "data": json.dumps(
                {
                    "phase": "ready",
                    "label": f"{ctx['model_name']} is loaded into inference",
                    "percent": 100,
                    "model_id": body.model_id,
                    "model_name": ctx["model_name"],
                    "backend": ctx["backend"],
                    "size_bytes": size_bytes,
                }
            ),
        }
        yield {
            "event": "complete",
            "data": json.dumps(
                {
                    "status": "loaded",
                    "backend": ctx["backend"],
                    "model_id": body.model_id,
                    "model_name": ctx["model_name"],
                    **status,
                }
            ),
        }

    return EventSourceResponse(event_gen())


async def _resolve_preload_context(
    db: Database,
    user_id: str,
    settings: ForgeSettings,
    model_id: str,
    inference_backend: str,
) -> dict[str, Any]:
    selected = await get_inference_option(
        db, user_id, model_id
    )
    if not selected:
        raise HTTPException(404, "Model not found in inventory")

    try:
        target = resolve_chat_target(
            selected,
            model_id=model_id,
            inference_backend=inference_backend,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    backend = target.get("inference_backend", inference_backend)

    path = target.get("model_path")
    if not path:
        path = await resolve_model_path(
            db,
            user_id,
            model_id=model_id,
            model_path=None,
            data_dir=settings.data_dir,
        )
    if not path:
        raise HTTPException(400, "Model path not found")

    from seiso.memory.protection import assess_path_memory_fit_for_load

    fit = assess_path_memory_fit_for_load(
        path,
        mode="chat",
        backend=backend,
    )
    if fit.get("memory_load_blocked"):
        raise HTTPException(
            400,
            fit.get("memory_load_blocked_reason")
            or "Model exceeds available memory on this machine",
        )

    payload = {
        "model_path": path,
        "model_format": target.get("model_format") or selected.get("format"),
        "inference_backend": backend,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
    }
    return {
        "payload": payload,
        "backend": backend,
        "model_name": selected.get("name") or model_id,
        "size_bytes": int(selected.get("size_bytes") or 0),
    }


def _warm_local_model(runner, payload: dict[str, Any]) -> None:
    model_path = payload["model_path"]
    route, resolved_path = runner._resolve_route(payload, model_path)
    pool = runner._pool
    pool.prepare_for_load(resolved_path, payload.get("inference_backend"))
    if route == "mlx":
        pool.get_mlx(resolved_path)
    elif route == "torch":
        pool.get_torch(resolved_path)
    else:
        from seiso.inference.tuning import estimate_llama_n_ctx

        messages = payload.get("messages") or []
        n_ctx = payload.get("n_ctx") or estimate_llama_n_ctx(
            messages,
            max_tokens=int(payload.get("max_tokens", 1)),
            model_path=resolved_path,
            model_format=payload.get("model_format"),
        )
        pool.get_llama(resolved_path, n_ctx=n_ctx)


async def _release_active_local_model(runner) -> None:
    pool = runner._pool
    if not pool.active_key:
        return
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, pool.cancel_and_unload)


def _active_local_model_would_change(pool, *, target_path: str, backend: str | None) -> bool:
    status = pool.status()
    if not status.get("active_model"):
        return False

    expected_pool_backend = _POOL_BACKEND_BY_API_BACKEND.get((backend or "").lower())
    if expected_pool_backend and status.get("backend") != expected_pool_backend:
        return True

    active_path = status.get("path")
    return bool(
        active_path and pool.normalize_path(active_path) != pool.normalize_path(target_path)
    )


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

    job_id = orchestrator.create_job(user_id=user_id)
    payload = body.model_dump(exclude={"messages"})
    payload["user_id"] = user_id

    knowledge_context: str | None = None
    if body.knowledge_base_id:
        from forge.api.routes.knowledge import _validate_kb_id

        kb_id = _validate_kb_id(body.knowledge_base_id)
        user_query = str(body.messages[-1].get("content", "")).strip() if body.messages else ""
        chunks = retrieve_knowledge_chunks(
            settings.data_dir,
            user_id=user_id,
            knowledge_base_id=kb_id,
            query=user_query,
        )
        knowledge_context = format_knowledge_context(chunks) or None

    trusted_messages, _user_content = await build_trusted_messages(
        db,
        thread_id=body.thread_id,
        client_messages=body.messages,
        persist_user=bool(body.thread_id),
        user_id=user_id,
        model_id=body.model_id,
        model_path=body.model_path,
        tools_enabled=body.tools,
        knowledge_context=knowledge_context,
    )
    payload["messages"] = trusted_messages

    if body.provider_id:
        prov = await db.get_provider(body.provider_id, user_id)
        if not prov:
            raise HTTPException(404, "Provider not found")
        if prov["provider_type"].lower() in {"openai", "anthropic"}:
            raise HTTPException(400, "Frontier cloud providers are not supported")
        payload["provider"] = {
            "provider_type": prov["provider_type"],
            "config": json.loads(prov["config_json"]),
        }
    if not body.provider_id:
        if body.model_id != ROUTER_MODEL_ID:
            _assert_inference_gpu_available()
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
            from seiso.memory.protection import assess_path_memory_fit_for_load

            fit = assess_path_memory_fit_for_load(path, mode="chat")
            if fit.get("memory_load_blocked"):
                raise HTTPException(
                    400,
                    fit.get("memory_load_blocked_reason")
                    or "Model exceeds available memory on this machine",
                )
            payload["model_path"] = path
            payload["inference_backend"] = body.inference_backend
        elif body.model_id:
            selected = await get_inference_option(db, user_id, body.model_id)
            if selected is None and settings.model_router_enabled and body.model_id == ROUTER_MODEL_ID:
                selected = next(
                    (
                        o
                        for o in await list_inference_options(
                            db,
                            user_id,
                            model_router_enabled=True,
                        )
                        if o["id"] == body.model_id
                    ),
                    None,
                )
            try:
                target = resolve_chat_target(
                    selected,
                    model_id=body.model_id,
                    inference_backend=body.inference_backend,
                )
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc

            payload["inference_backend"] = target.get("inference_backend", body.inference_backend)
            payload["model_format"] = target.get("model_format")

            if target.get("inference_backend") == BACKEND_ROUTER:
                if not settings.model_router_enabled:
                    raise HTTPException(
                        400,
                        "Smart Router is not enabled (set SEISO_MODEL_ROUTER_ENABLED=1)",
                    )
                payload["use_model_router"] = True
                payload["router_model"] = body.router_model
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
                from seiso.memory.protection import assess_path_memory_fit_for_load

                fit = assess_path_memory_fit_for_load(
                    path,
                    mode="chat",
                    backend=payload.get("inference_backend"),
                )
                if fit.get("memory_load_blocked"):
                    raise HTTPException(
                        400,
                        fit.get("memory_load_blocked_reason")
                        or "Model exceeds available memory on this machine",
                    )
                payload["model_path"] = path
        else:
            raise HTTPException(400, "Select a model from inventory or provide model_path")

    if body.draft_model_id and body.draft_model_path:
        raise HTTPException(403, "Provide draft_model_id or draft_model_path, not both")

    if body.draft_model_id or body.draft_model_path:
        if body.provider_id:
            raise HTTPException(400, "Speculative decoding is not available for cloud providers")
        if body.draft_model_path:
            draft_path = await resolve_model_path(
                db,
                user_id,
                model_id=None,
                model_path=body.draft_model_path,
                data_dir=settings.data_dir,
            )
        else:
            draft_selected = await get_inference_option(
                db, user_id, body.draft_model_id
            )
            if not draft_selected:
                raise HTTPException(404, "Draft model not found")
            draft_path = draft_selected.get("path")
            if not draft_path:
                raise HTTPException(400, "Draft model must be a local safetensors/checkpoint path")
        if not draft_path:
            raise HTTPException(400, "Invalid draft model path")
        from seiso.memory.protection import assess_path_memory_fit_for_load

        draft_fit = assess_path_memory_fit_for_load(draft_path, mode="chat", backend=BACKEND_TORCH)
        if draft_fit.get("memory_load_blocked"):
            raise HTTPException(
                400,
                draft_fit.get("memory_load_blocked_reason")
                or "Draft model exceeds available memory on this machine",
            )
        payload["draft_model_path"] = draft_path
        payload["inference_backend"] = BACKEND_TORCH

    if body.stream:
        use_router = bool(payload.get("use_model_router"))
        can_stream_router = use_router and not body.tools and not body.provider_id
        can_stream_local = not body.tools and not body.provider_id and not use_router

        async def event_gen():
            if can_stream_router:
                parts: list[str] = []
                try:
                    orchestrator._emit_log(job_id, "Streaming inference (smart router)")
                    async for token in orchestrator.stream_router(payload):
                        parts.append(token)
                        yield {"event": "token", "data": token}
                    content = "".join(parts)
                    if body.thread_id:
                        await db.add_message(body.thread_id, "assistant", content)
                    yield {"event": "message", "data": content}
                    yield {"event": "done", "data": job_id}
                except Exception as exc:
                    yield {"event": "error", "data": str(exc)}
                return

            if can_stream_local:
                streamed: list[str] = []
                raw_parts: list[str] = []
                sanitizer = StreamingOutputSanitizer(strip_tool_calls=not body.tools)
                backend_label = (
                    "speculative"
                    if payload.get("draft_model_path")
                    else (payload.get("inference_backend") or "local")
                )
                cancelled = False
                try:
                    orchestrator._emit_log(job_id, f"Streaming inference ({backend_label})")
                    async for token in orchestrator.stream_local(payload):
                        raw_parts.append(token)
                        for chunk in sanitizer.feed(token):
                            streamed.append(chunk)
                            yield {"event": "token", "data": chunk}
                    for chunk in sanitizer.finish():
                        streamed.append(chunk)
                        yield {"event": "token", "data": chunk}
                    content = sanitize_llm_output(
                        "".join(raw_parts), strip_tool_calls=not body.tools
                    )
                    if body.thread_id:
                        await db.add_message(body.thread_id, "assistant", content)
                    yield {"event": "message", "data": content}
                    yield {"event": "done", "data": job_id}
                except Exception as exc:
                    yield {"event": "error", "data": str(exc)}
                except asyncio.CancelledError:
                    cancelled = True
                    raise
                finally:
                    if cancelled:
                        await orchestrator._runner.cancel_generation()
                return

            await orchestrator.start(job_id, payload)
            async for line in orchestrator.stream_logs(job_id):
                yield {"event": "log", "data": line}
            job = await orchestrator.wait_for(job_id)
            if job and job.status.value == "failed":
                yield {"event": "error", "data": job.error or "Inference failed"}
            elif job and job.result.get("content"):
                content = sanitize_llm_output(
                    job.result["content"],
                    strip_tool_calls=not body.tools,
                )
                if body.thread_id:
                    await db.add_message(body.thread_id, "assistant", content)
                yield {"event": "message", "data": content}
            yield {"event": "done", "data": job_id}

        return EventSourceResponse(event_gen())

    await orchestrator.start(job_id, payload)
    job = await orchestrator.wait_for(job_id)
    if not job:
        raise HTTPException(500, "Job lost")
    if job.status.value == "failed":
        raise HTTPException(500, job.error or "Inference failed")
    if body.thread_id and job.result.get("content"):
        content = sanitize_llm_output(job.result["content"], strip_tool_calls=not body.tools)
        await db.add_message(body.thread_id, "assistant", content)
    result = dict(job.result)
    if result.get("content") and not body.tools:
        result["content"] = sanitize_llm_output(result["content"], strip_tool_calls=True)
    return {"job_id": job_id, **result}
