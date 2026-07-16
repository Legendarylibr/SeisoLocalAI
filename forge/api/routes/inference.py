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
from seiso.inference.backends import BACKEND_LLAMASWAP

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
    return await db.get_messages(thread_id, user_id)


@router.get("/context")
async def get_chat_context(
    user_id: Annotated[str, Depends(get_current_user_id)],
    db: Annotated[Database, Depends(get_db)],
    settings: Annotated[ForgeSettings, Depends(get_settings)],
    thread_id: str | None = None,
    max_tokens: int = Query(default=2048, ge=1, le=131072),
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
        history = await db.get_messages(thread_id, user_id)
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
        chunks = await asyncio.to_thread(
            retrieve_knowledge_chunks,
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

    _begin_generation_or_raise(orchestrator, user_id)
    try:
        await orchestrator.preload_model(ctx["payload"])
        status = orchestrator.inference_status()
        pinned = ctx["payload"].get("sidecar_num_ctx") or ctx["payload"].get("n_ctx")
        if pinned is None:
            pinned = status.get("n_ctx")
        runtime = orchestrator.inference_runtime_stats()
        return {
            "status": "loaded",
            "resident_status": (
                "loaded" if runtime.get("resident_confirmed", True) else "sidecar-ready"
            ),
            "backend": ctx["backend"],
            "n_ctx": pinned,
            "sidecar_num_ctx": ctx["payload"].get("sidecar_num_ctx") or pinned,
            "runtime": runtime,
            **status,
        }
    except asyncio.CancelledError:
        await orchestrator.cancel_and_unload_for_user(user_id)
        raise
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

    target_path = ctx["payload"]["model_path"]
    size_bytes = ctx.get("size_bytes", 0)
    eta = estimate_load_eta_seconds(size_bytes)

    async def event_gen():
        try:
            _begin_generation_or_raise(orchestrator, user_id)
            switching = orchestrator.would_switch_model(target_path, ctx["backend"])
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
                await orchestrator.prepare_model_for_load(target_path, ctx["backend"])

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
            await orchestrator.preload_model(ctx["payload"])
        except asyncio.CancelledError:
            await orchestrator.cancel_and_unload_for_user(user_id)
            raise
        except Exception as exc:
            yield {"event": "error", "data": str(exc)}
            return
        finally:
            orchestrator.end_generation_for_user(user_id)

        status = orchestrator.inference_status()
        pinned = ctx["payload"].get("sidecar_num_ctx") or ctx["payload"].get("n_ctx")
        if pinned is None:
            pinned = status.get("n_ctx")
        runtime = orchestrator.inference_runtime_stats()
        resident_confirmed = runtime.get("resident_confirmed", True)
        yield {
            "event": "progress",
            "data": json.dumps(
                {
                    "phase": "ready",
                    "label": (
                        f"{ctx['model_name']} is loaded into inference"
                        if resident_confirmed
                        else f"{ctx['model_name']} sidecar is ready"
                    ),
                    "percent": 100,
                    "model_id": body.model_id,
                    "model_name": ctx["model_name"],
                    "backend": ctx["backend"],
                    "size_bytes": size_bytes,
                    "n_ctx": pinned,
                }
            ),
        }
        yield {
            "event": "complete",
            "data": json.dumps(
                {
                    "status": "loaded",
                    "resident_status": ("loaded" if resident_confirmed else "sidecar-ready"),
                    "backend": ctx["backend"],
                    "model_id": body.model_id,
                    "model_name": ctx["model_name"],
                    "n_ctx": pinned,
                    "sidecar_num_ctx": ctx["payload"].get("sidecar_num_ctx") or pinned,
                    "runtime": runtime,
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
    if body.knowledge_base_id:
        # Resolve the latest user turn once for retrieval, then build messages
        # with knowledge injected (avoids a full trusted-message rebuild).
        last = body.messages[-1] if body.messages else {}
        user_query = normalize_text(str(last.get("content") or "")).strip()
        kb_id = validate_kb_id(body.knowledge_base_id)
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
        from forge.providers.router import LOCAL_PROVIDER_TYPES

        ptype = prov["provider_type"].lower()
        if ptype not in LOCAL_PROVIDER_TYPES:
            raise HTTPException(
                400, f"Unsupported chat provider type: {prov['provider_type']}"
            )
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
        # Reuse preload-pinned sidecar KV size across turns (avoid re-bucket reload).
        if (
            payload.get("inference_backend") == BACKEND_LLAMASWAP
            and payload.get("model_path")
            and payload.get("sidecar_num_ctx") is None
        ):
            pinned = orchestrator.pinned_inference_context(str(payload["model_path"]))
            if pinned is not None:
                payload["sidecar_num_ctx"] = pinned
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
        job_id = (
            str(uuid.uuid4())
            if (can_stream_router or can_stream_local)
            else orchestrator.create_job(user_id=user_id)
        )
        _begin_generation_or_raise(orchestrator, user_id)

        async def event_gen():
            if can_stream_router:
                parts: list[str] = []
                output_tokens = 0
                try:
                    orchestrator.emit_log(job_id, "Streaming inference (smart router)")
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
                from forge.services.generation_continue import (
                    authoritative_pass_tokens,
                    build_continue_messages,
                    can_schedule_another_continue,
                    effective_pass_tokens,
                    hit_length_limit,
                    looks_incomplete_reply,
                    next_pass_max_tokens,
                    reply_still_truncated,
                    resolve_auto_continue_limits,
                    resolve_finish_reason,
                    should_auto_continue,
                )
                from seiso.memory.protection import sanitize_inference_payload

                streamed: list[str] = []
                raw_parts: list[str] = []
                sanitizer = StreamingOutputSanitizer(strip_tool_calls=not body.tools)
                backend_label = (
                    "speculative"
                    if payload.get("draft_model_path")
                    else (payload.get("inference_backend") or "local")
                )
                cancelled = False
                # Keep n_ctx fixed across auto-continues — never grow KV for long replies.
                fixed_n_ctx = payload.get("n_ctx")
                base_messages = list(payload.get("messages") or [])
                isolated = (
                    str(payload.get("inference_backend") or "").lower() == "llamaswap"
                )
                # Client max_tokens is the desired overall reply length; per-pass
                # generation is clamped separately for OOM safety and chunked via
                # auto-continue until this total (or a dynamic long-form budget) is met.
                requested_max_tokens = max(1, int(body.max_tokens or 2048))
                pass_payload = dict(payload)
                if fixed_n_ctx is not None:
                    pass_payload["n_ctx"] = fixed_n_ctx
                    pass_payload["pin_n_ctx"] = True
                effective = sanitize_inference_payload(
                    pass_payload, isolated=isolated
                )
                pass_max_tokens = max(1, int(effective.get("max_tokens") or 2048))
                # Sidecar applies its own native completion cap; mirror it here so
                # length detection / auto-continue match what Ollama actually generates.
                if isolated:
                    try:
                        from seiso.inference.llamaswap import sidecar_max_tokens

                        pass_max_tokens = max(1, int(sidecar_max_tokens(pass_max_tokens)))
                    except Exception:
                        pass
                pass_payload["max_tokens"] = pass_max_tokens
                if fixed_n_ctx is not None:
                    # Re-apply pin after sanitize (pin flag is consumed there).
                    pass_payload["n_ctx"] = fixed_n_ctx
                    pass_payload["pin_n_ctx"] = True
                continues_used = 0
                total_output_tokens = 0
                last_pass_tokens = 0
                # Accumulated text from the latest generation chunk (not a secret).
                last_reply_chunk = ""
                finish_reason = "stop"
                # One free retry when a continue pass returns empty after a cut-off.
                empty_continue_retries = 0
                # Incomplete-only continues (mid-sentence EOS) without a length hit.
                # Cap progress-free streaks so a model that always ends mid-line
                # cannot spin until the full multi-pass budget is wasted.
                incomplete_only_streak = 0
                low_progress_streak = 0
                _MAX_INCOMPLETE_ONLY_STREAK = 16
                _MAX_LOW_PROGRESS_STREAK = 2
                headroom_mb: float | None = None
                try:
                    from seiso.memory.protection import headroom_mb as _headroom_mb

                    headroom_mb = float(_headroom_mb())
                except Exception:
                    headroom_mb = None
                # Capture the OOM-safe per-pass budget once; continues must not grow it.
                # Total/continues scale with request + long-form intent; VRAM stays flat
                # via fixed n_ctx + prompt trim.
                base_pass_max_tokens = pass_max_tokens
                max_continues, total_budget = resolve_auto_continue_limits(
                    requested_max_tokens=requested_max_tokens,
                    pass_max_tokens=base_pass_max_tokens,
                    messages=base_messages,
                    headroom_mb=headroom_mb,
                )
                try:
                    orchestrator.emit_log(
                        job_id,
                        f"Streaming inference ({backend_label}; "
                        f"pass≤{base_pass_max_tokens} tok, total≤{total_budget} tok, "
                        f"continues≤{max_continues})",
                    )
                    while True:
                        pass_raw: list[str] = []
                        pass_tokens = 0
                        pass_finish: str | None = None
                        last_meta: dict[str, Any] = {}
                        async for update in orchestrator.stream_local_updates(
                            pass_payload
                        ):
                            meta = dict(update.metadata or {})
                            last_meta = meta
                            reason = meta.get("finish_reason")
                            if isinstance(reason, str) and reason:
                                pass_finish = reason
                            pass_tokens = max(pass_tokens, int(update.output_tokens))
                            # Prefer Ollama eval_count / similar over stream estimates.
                            pass_tokens = authoritative_pass_tokens(pass_tokens, meta)
                            stats_payload = {
                                "output_tokens": total_output_tokens + pass_tokens,
                                "auto_continues": continues_used,
                                **meta,
                            }
                            if not update.text:
                                if pass_finish:
                                    stats_payload["finish_reason"] = pass_finish
                                yield {
                                    "event": "stats",
                                    "data": json.dumps(stats_payload),
                                }
                                continue
                            pass_raw.append(update.text)
                            raw_parts.append(update.text)
                            yield {
                                "event": "stats",
                                "data": json.dumps(stats_payload),
                            }
                            for chunk in sanitizer.feed(update.text):
                                streamed.append(chunk)
                                yield {"event": "token", "data": chunk}

                        pass_text = "".join(pass_raw)
                        last_reply_chunk = pass_text
                        pass_tokens = effective_pass_tokens(
                            pass_tokens,
                            pass_text=pass_text,
                            metadata=last_meta,
                        )
                        total_output_tokens += pass_tokens
                        last_pass_tokens = pass_tokens
                        draft_so_far = sanitize_llm_output(
                            "".join(raw_parts), strip_tool_calls=not body.tools
                        )
                        hit_length = hit_length_limit(
                            pass_tokens,
                            pass_max_tokens,
                            finish_reason=pass_finish,
                            pass_text=pass_text,
                            metadata=last_meta,
                        )
                        incomplete = looks_incomplete_reply(
                            pass_text if pass_text.strip() else draft_so_far
                        )
                        # Prefer "length" when the pass is full so auto-continue
                        # fires even if the backend mislabels the stop reason.
                        if hit_length:
                            finish_reason = "length"
                        else:
                            finish_reason = resolve_finish_reason(
                                hit_length=False,
                                explicit=pass_finish,
                            )

                        # Empty continue pass after a cut-off: retry once with a
                        # stronger continue cue (live Qwen songs often EOS empty).
                        force_retry_empty = False
                        if (
                            not pass_text.strip()
                            and continues_used > 0
                            and empty_continue_retries < 1
                            and looks_incomplete_reply(draft_so_far)
                            and can_schedule_another_continue(
                                continues_used=continues_used,
                                max_continues=max_continues,
                                total_output_tokens=total_output_tokens,
                                total_budget=total_budget,
                                pass_max_tokens=base_pass_max_tokens,
                            )
                        ):
                            force_retry_empty = True
                            empty_continue_retries += 1

                        want_continue = force_retry_empty or should_auto_continue(
                            pass_output_tokens=pass_tokens,
                            max_tokens=pass_max_tokens,
                            pass_text=pass_text if pass_text.strip() else draft_so_far,
                            continues_used=continues_used,
                            max_continues=max_continues,
                            finish_reason=finish_reason,
                            cancelled=cancelled,
                            total_output_tokens=total_output_tokens,
                            total_budget=total_budget,
                            metadata=last_meta,
                            force_incomplete=force_retry_empty or (
                                incomplete and not pass_text.strip()
                            ),
                        )
                        # Safety rails for incomplete-only continues (no length hit):
                        # stop after many incomplete passes or tiny progress repeats.
                        if want_continue and incomplete and not hit_length and not force_retry_empty:
                            incomplete_only_streak += 1
                            if pass_tokens < 16 and continues_used > 0:
                                low_progress_streak += 1
                            else:
                                low_progress_streak = 0
                            if (
                                incomplete_only_streak > _MAX_INCOMPLETE_ONLY_STREAK
                                or low_progress_streak >= _MAX_LOW_PROGRESS_STREAK
                            ):
                                want_continue = False
                        elif hit_length:
                            incomplete_only_streak = 0
                            low_progress_streak = 0
                        if not want_continue:
                            break

                        continues_used += 1
                        partial = draft_so_far
                        # Next chunk: never exceed remaining multi-pass budget.
                        chunk_tokens = next_pass_max_tokens(
                            base_pass_max_tokens=base_pass_max_tokens,
                            total_output_tokens=total_output_tokens,
                            total_budget=total_budget,
                        )
                        if chunk_tokens < 8:
                            finish_reason = "length"
                            break
                        use_strong = (
                            force_retry_empty
                            or incomplete
                            or hit_length
                            or empty_continue_retries > 0
                        )
                        reason_label = (
                            "empty-retry"
                            if force_retry_empty
                            else "incomplete"
                            if incomplete and not hit_length
                            else "length"
                        )
                        orchestrator.emit_log(
                            job_id,
                            f"Auto-continuing {reason_label} reply "
                            f"(pass {continues_used + 1}, budget {chunk_tokens} tok, "
                            f"total {total_output_tokens}/{total_budget})",
                        )
                        yield {
                            "event": "log",
                            "data": (
                                f"Reply incomplete — continuing "
                                f"({continues_used}/{max_continues})…"
                            ),
                        }
                        # Reuse fixed n_ctx. Trim the growing transcript into that
                        # window instead of raising n_ctx (OOM).
                        pass_payload = {
                            **payload,
                            "messages": build_continue_messages(
                                base_messages,
                                partial,
                                n_ctx=int(fixed_n_ctx) if fixed_n_ctx is not None else None,
                                max_tokens=chunk_tokens,
                                strong=use_strong,
                            ),
                            "max_tokens": chunk_tokens,
                        }
                        if fixed_n_ctx is not None:
                            pass_payload["n_ctx"] = fixed_n_ctx
                            pass_payload["pin_n_ctx"] = True
                        continued = sanitize_inference_payload(
                            pass_payload, isolated=isolated
                        )
                        # Never raise the per-pass budget above the first-pass OOM-safe
                        # cap or the remaining multi-pass allowance.
                        pass_max_tokens = max(
                            1,
                            min(
                                chunk_tokens,
                                base_pass_max_tokens,
                                int(continued.get("max_tokens") or chunk_tokens),
                            ),
                        )
                        pass_payload["max_tokens"] = pass_max_tokens
                        if fixed_n_ctx is not None:
                            pass_payload["n_ctx"] = fixed_n_ctx
                            pass_payload["pin_n_ctx"] = True
                        if pass_max_tokens < 8:
                            finish_reason = "length"
                            break

                    for chunk in sanitizer.finish():
                        streamed.append(chunk)
                        yield {"event": "token", "data": chunk}
                    content = sanitize_llm_output(
                        "".join(raw_parts), strip_tool_calls=not body.tools
                    )
                    still_truncated = reply_still_truncated(
                        last_pass_tokens=last_pass_tokens,
                        pass_max_tokens=pass_max_tokens,
                        finish_reason=finish_reason,
                        total_output_tokens=total_output_tokens,
                        total_budget=total_budget,
                        continues_used=continues_used,
                        max_continues=max_continues,
                        pass_text=last_reply_chunk,
                        metadata=last_meta,
                        cancelled=cancelled,
                    )
                    if still_truncated:
                        finish_reason = "length"
                    if body.thread_id:
                        await db.add_message(
                            body.thread_id,
                            "assistant",
                            content,
                            metadata={
                                "truncated": still_truncated,
                                "auto_continues": continues_used,
                                "finish_reason": finish_reason,
                                "output_tokens": total_output_tokens,
                            },
                        )
                    final_stats = {
                        "output_tokens": total_output_tokens,
                        "finish_reason": finish_reason,
                        "auto_continues": continues_used,
                        "truncated": still_truncated,
                        "total_budget": total_budget,
                        **last_meta,
                    }

                    final_stats["finish_reason"] = finish_reason
                    final_stats["truncated"] = still_truncated
                    final_stats["auto_continues"] = continues_used
                    yield {
                        "event": "stats",
                        "data": json.dumps(final_stats),
                    }
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
