"""Chat and inference routes with SSE streaming."""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sse_starlette.sse import EventSourceResponse

from forge.api.deps import get_db, get_inference_orchestrator
from forge.api.routes._inference_common import (
    _assert_inference_gpu_available,
    _begin_generation_or_raise,
)
from forge.api.schemas.inference import ChatRequest, PreloadRequest, ThreadCreate
from forge.config import ForgeSettings, get_settings
from forge.db.store import Database
from forge.orchestrators.inference import InferenceOrchestrator
from forge.security.auth import get_current_user_id
from forge.services.chat_messages import build_trusted_messages
from forge.services.download_progress import estimate_load_eta_seconds
from forge.services.hardware import hardware_profile
from forge.services.hf_cache_sync import schedule_hf_cache_inventory_sync
from forge.services.inference_chat import (
    prepare_local_chat_target,
    resolve_draft_model,
    resolve_preload_context,
)
from forge.services.inference_models import get_inference_option, list_inference_options
from forge.services.knowledge_context import (
    format_knowledge_context,
    retrieve_knowledge_chunks,
)
from forge.services.knowledge_paths import validate_kb_id
from forge.services.llm_output import StreamingOutputSanitizer, sanitize_llm_output
from forge.services.model_router_client import ROUTER_MODEL_ID, fetch_router_status
from forge.tools.sanitize import normalize_text

router = APIRouter(prefix="/inference", tags=["inference"])


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
    sync_cache: bool = Query(
        False,
        description="Synchronously refresh Hugging Face cache inventory before returning.",
    ),
) -> dict[str, Any]:
    """Unified model dropdown: HF Hub inventory, CLI paths, fine-tune/export outputs."""
    from forge.services.hardware import hardware_summary

    await schedule_hf_cache_inventory_sync(
        db,
        user_id,
        data_dir=settings.data_dir,
        hf_cache_dir=settings.hf_cache_dir,
        sync_cache=sync_cache,
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
    from forge.services.hf_auth import resolve_hf_token_for_download
    from forge.services.inference_variants import get_model_variants

    hf_token, _ = resolve_hf_token_for_download(
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
    user_id: Annotated[str, Depends(get_current_user_id)],
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
    draft_message: str | None = None,
) -> dict[str, Any]:
    """Context window usage for the chat UI."""
    from forge.services.chat_context import context_status_for_history
    from forge.services.knowledge_context import (
        format_knowledge_context,
        retrieve_knowledge_chunks,
    )

    history: list[dict] = []
    if thread_id:
        if not await db.get_thread_for_user(thread_id, user_id):
            raise HTTPException(404, "Thread not found")
        history = await db.get_messages(thread_id)
    draft_content = normalize_text(draft_message or "").strip()
    if draft_content:
        draft_matches_last = (
            history
            and history[-1].get("role") == "user"
            and history[-1].get("content") == draft_content
        )
        if not draft_matches_last:
            history = [*history, {"role": "user", "content": draft_content}]

    model_path: str | None = None
    model_format: str | None = None
    model_name: str | None = None
    if model_id:
        target = await prepare_local_chat_target(
            db,
            user_id,
            settings,
            model_id=model_id,
            model_router_enabled=settings.model_router_enabled,
            check_memory=False,
            sanitize=False,
        )
        if target.get("use_model_router"):
            model_path = None
            model_format = None
            model_name = "Smart Router"
        else:
            model_path = target.get("model_path")
            model_format = target.get("model_format")
            model_name = target.get("model_name")

    knowledge_context: str | None = None
    if knowledge_base_id:
        kb_id = validate_kb_id(knowledge_base_id)
        last_user = ""
        for msg in reversed(history):
            if msg.get("role") == "user":
                last_user = normalize_text(str(msg.get("content", ""))).strip()
                break
        chunks = retrieve_knowledge_chunks(
            settings.data_dir,
            user_id=user_id,
            knowledge_base_id=kb_id,
            query=last_user,
        )
        knowledge_context = format_knowledge_context(chunks, knowledge_base_id=kb_id) or None

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
    try:
        orchestrator.assert_generation_available_for_user(user_id)
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc
    ctx = await resolve_preload_context(
        db,
        user_id,
        settings,
        body.model_id,
        body.inference_backend,
        max_tokens=body.max_tokens,
        n_ctx=body.n_ctx,
    )

    loop = asyncio.get_running_loop()
    _begin_generation_or_raise(orchestrator, user_id)
    try:
        await loop.run_in_executor(None, lambda: orchestrator._runner.warm_model(ctx["payload"]))
        status = orchestrator._runner.pool.status()
        return {"status": "loaded", "backend": ctx["backend"], **status}
    finally:
        orchestrator.end_generation_for_user(user_id)


@router.post("/preload/stream")
async def preload_model_stream(
    body: PreloadRequest,
    user_id: Annotated[str, Depends(get_current_user_id)],
    db: Annotated[Database, Depends(get_db)],
    orchestrator: Annotated[InferenceOrchestrator, Depends(get_inference_orchestrator)],
    settings: Annotated[ForgeSettings, Depends(get_settings)],
):
    _assert_inference_gpu_available()
    try:
        orchestrator.assert_generation_available_for_user(user_id)
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc
    ctx = await resolve_preload_context(
        db,
        user_id,
        settings,
        body.model_id,
        body.inference_backend,
        max_tokens=body.max_tokens,
        n_ctx=body.n_ctx,
    )

    runner = orchestrator._runner
    pool = runner.pool
    target_path = ctx["payload"]["model_path"]
    size_bytes = ctx.get("size_bytes", 0)
    eta = estimate_load_eta_seconds(size_bytes)
    loop = asyncio.get_running_loop()

    async def event_gen():
        try:
            _begin_generation_or_raise(orchestrator, user_id)
            switching = runner.pool.would_switch_model(target_path, ctx["backend"])
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
                    lambda: runner.pool.prepare_for_load(target_path, ctx["backend"]),
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
            await loop.run_in_executor(None, lambda: runner.warm_model(ctx["payload"]))
        except asyncio.CancelledError:
            await orchestrator.cancel_and_unload_for_user(user_id)
            raise
        except Exception as exc:
            yield {"event": "error", "data": str(exc)}
            return
        finally:
            orchestrator.end_generation_for_user(user_id)

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

    payload = body.model_dump(exclude={"messages"})
    payload["user_id"] = user_id

    knowledge_context: str | None = None
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
    if body.knowledge_base_id:
        kb_id = validate_kb_id(body.knowledge_base_id)
        user_query = normalize_text(_user_content or "").strip()
        chunks = retrieve_knowledge_chunks(
            settings.data_dir,
            user_id=user_id,
            knowledge_base_id=kb_id,
            query=user_query,
        )
        knowledge_context = format_knowledge_context(chunks, knowledge_base_id=kb_id) or None
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
        model_updates = await prepare_local_chat_target(
            db,
            user_id,
            settings,
            model_id=body.model_id,
            model_path=body.model_path,
            inference_backend=body.inference_backend,
            model_router_enabled=settings.model_router_enabled,
            max_tokens=body.max_tokens,
            n_ctx=body.n_ctx,
            messages=trusted_messages,
            check_memory=True,
            sanitize=True,
        )
        # Runner also sanitizes; keep n_ctx/max_tokens aligned with preload.
        for key in (
            "model_path",
            "inference_backend",
            "model_format",
            "model_metadata",
            "use_model_router",
            "max_tokens",
            "n_ctx",
        ):
            if key in model_updates and model_updates[key] is not None:
                payload[key] = model_updates[key]
        if model_updates.get("use_model_router"):
            payload["router_model"] = body.router_model

    if body.draft_model_id or body.draft_model_path:
        if body.provider_id:
            raise HTTPException(400, "Speculative decoding is not available for cloud providers")
        payload.update(
            await resolve_draft_model(
                db,
                user_id,
                settings,
                draft_model_id=body.draft_model_id,
                draft_model_path=body.draft_model_path,
                target_model_path=payload.get("model_path"),
            )
        )

    if body.stream:
        use_router = bool(payload.get("use_model_router"))
        can_stream_router = use_router and not body.tools and not body.provider_id
        can_stream_local = not body.tools and not body.provider_id and not use_router
        job_id = str(uuid.uuid4()) if (can_stream_router or can_stream_local) else orchestrator.create_job(user_id=user_id)
        _begin_generation_or_raise(orchestrator, user_id)

        async def event_gen():
            if can_stream_router:
                parts: list[str] = []
                output_tokens = 0
                try:
                    orchestrator._emit_log(job_id, "Streaming inference (smart router)")
                    async for token in orchestrator.stream_router(payload):
                        parts.append(token)
                        output_tokens += 1
                        yield {
                            "event": "stats",
                            "data": json.dumps({"output_tokens": output_tokens}),
                        }
                        yield {"event": "token", "data": token}
                    content = "".join(parts)
                    if body.thread_id:
                        await db.add_message(body.thread_id, "assistant", content)
                    yield {"event": "message", "data": content}
                    yield {"event": "done", "data": job_id}
                except asyncio.CancelledError:
                    await orchestrator.cancel_generation_for_user(user_id)
                    raise
                except Exception as exc:
                    await orchestrator.cancel_generation_for_user(user_id)
                    yield {"event": "error", "data": str(exc)}
                finally:
                    orchestrator.end_generation_for_user(user_id)
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
                    async for update in orchestrator.stream_local_updates(payload):
                        raw_parts.append(update.text)
                        yield {
                            "event": "stats",
                            "data": json.dumps({"output_tokens": update.output_tokens}),
                        }
                        for chunk in sanitizer.feed(update.text):
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
                except asyncio.CancelledError:
                    cancelled = True
                    raise
                except Exception as exc:
                    await orchestrator.cancel_generation_for_user(user_id)
                    yield {"event": "error", "data": str(exc)}
                finally:
                    if cancelled:
                        await orchestrator.cancel_generation_for_user(user_id)
                    else:
                        orchestrator.end_generation_for_user(user_id)
                return

            try:
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
            except asyncio.CancelledError:
                await orchestrator.cancel_generation_for_user(user_id)
                raise
            except Exception as exc:
                await orchestrator.cancel_generation_for_user(user_id)
                yield {"event": "error", "data": str(exc)}

        return EventSourceResponse(event_gen())

    job_id = orchestrator.create_job(user_id=user_id)
    _begin_generation_or_raise(orchestrator, user_id)
    try:
        await orchestrator.start(job_id, payload)
        job = await orchestrator.wait_for(job_id)
    except Exception:
        await orchestrator.cancel_generation_for_user(user_id)
        raise
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
