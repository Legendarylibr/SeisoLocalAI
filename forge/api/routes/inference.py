"""Chat and inference routes with SSE streaming."""

from __future__ import annotations

import asyncio
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
from forge.services.chat_messages import build_trusted_messages
from forge.services.download_progress import estimate_load_eta_seconds
from forge.services.hardware import hardware_profile
from forge.services.hf_cache_inventory import sync_hf_cache_inventory
from forge.services.inference_models import list_inference_options, resolve_chat_target
from forge.services.llm_output import (
    StreamingOutputSanitizer,
    chunk_sanitized_output,
    sanitize_llm_output,
)
from forge.services.models import resolve_model_path
from seiso.inference.backends import BACKEND_LLAMACPP, BACKEND_MLX, BACKEND_OLLAMA, BACKEND_TORCH

router = APIRouter(prefix="/inference", tags=["inference"])

_POOL_BACKEND_BY_API_BACKEND = {
    BACKEND_LLAMACPP: "llama",
    "llama": "llama",
    BACKEND_MLX: "mlx",
    BACKEND_TORCH: "torch",
}


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
    provider_id: str | None = None
    defense: bool | None = Field(
        default=None,
        description="Enable AutoDefense scan (requires SEISO_AUTODEFENSE_ENABLED). null=on when server enabled.",
    )


class ThreadCreate(BaseModel):
    title: str = "New chat"
    model_id: str | None = None


class PreloadRequest(BaseModel):
    model_id: str
    inference_backend: str = Field(default="auto", description="auto | llamacpp | ollama | mlx | torch")


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
        ollama_base_url=settings.ollama_base_url,
        profile=profile,
    )
    return {
        "models": options,
        "total": len(options),
        "hardware_summary": hardware_summary(profile),
        "preferred_inference_backend": profile.get("preferred_inference_backend"),
        "local_only": True,
    }


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
    ctx = await _resolve_preload_context(db, user_id, settings, body.model_id, body.inference_backend)
    if ctx.get("ollama_only"):
        await _release_active_local_model(orchestrator._runner)
        return ctx["response"]

    loop = asyncio.get_running_loop()
    await orchestrator.release_ollama_model()
    await loop.run_in_executor(None, lambda: _warm_local_model(orchestrator._runner, ctx["payload"]))
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
    ctx = await _resolve_preload_context(db, user_id, settings, body.model_id, body.inference_backend)
    if ctx.get("ollama_only"):
        ollama_model = ctx["response"].get("ollama_model") or ctx["response"].get("active_model")
        size_bytes = int(ctx.get("size_bytes") or 0)
        eta = estimate_load_eta_seconds(size_bytes) if size_bytes else 8
        runner = orchestrator._runner
        pool = runner._pool
        loop = asyncio.get_running_loop()

        async def ollama_gen():
            from forge.providers.ollama import warm_model

            switching_ollama = bool(
                orchestrator.active_ollama_model
                and orchestrator.active_ollama_model != ollama_model
            )
            if pool.active_key or switching_ollama:
                yield {
                    "event": "progress",
                    "data": json.dumps(
                        {
                            "phase": "unloading",
                            "label": "Releasing local model from VRAM before Ollama load",
                            "percent": 5,
                            "eta_seconds": 2,
                        }
                    ),
                }
            await orchestrator.prepare_ollama_model(ollama_model, settings.ollama_base_url)

            yield {
                "event": "progress",
                "data": json.dumps(
                    {
                        "phase": "loading",
                        "label": f"Loading {ollama_model} in Ollama",
                        "percent": 20,
                        "eta_seconds": eta,
                        "model_name": ollama_model,
                        "backend": BACKEND_OLLAMA,
                        "size_bytes": size_bytes,
                    }
                ),
            }
            try:
                await warm_model(ollama_model, settings.ollama_base_url)
            except Exception as exc:
                yield {"event": "error", "data": str(exc)}
                return
            yield {
                "event": "complete",
                "data": json.dumps(
                    {
                        **ctx["response"],
                        "status": "loaded",
                        "backend": BACKEND_OLLAMA,
                        "model_name": ollama_model,
                    }
                ),
            }

        return EventSourceResponse(ollama_gen())

    runner = orchestrator._runner
    pool = runner._pool
    target_path = ctx["payload"]["model_path"]
    size_bytes = ctx.get("size_bytes", 0)
    eta = estimate_load_eta_seconds(size_bytes)
    loop = asyncio.get_running_loop()

    async def event_gen():
        if orchestrator.active_ollama_model:
            yield {
                "event": "progress",
                "data": json.dumps(
                    {
                        "phase": "unloading",
                        "label": "Releasing Ollama model from VRAM",
                        "percent": 5,
                        "eta_seconds": 2,
                    }
                ),
            }
            await orchestrator.release_ollama_model()

        switching = _active_local_model_would_change(pool, target_path=target_path, backend=ctx["backend"])
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
            await loop.run_in_executor(None, pool.cancel_and_unload)

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
    options = await list_inference_options(db, user_id, ollama_base_url=settings.ollama_base_url)
    selected = next((o for o in options if o["id"] == model_id), None)
    if not selected:
        raise HTTPException(404, "Model not found in inventory")

    if selected.get("memory_load_blocked"):
        raise HTTPException(
            400,
            selected.get("memory_load_blocked_reason") or "Model exceeds available memory on this machine",
        )

    if selected.get("kind") == "ollama":
        response = {
            "status": "ready",
            "backend": BACKEND_OLLAMA,
            "ollama_model": selected.get("ollama_model"),
            "active_model": selected.get("ollama_model"),
        }
        return {
            "ollama_only": True,
            "response": response,
            "size_bytes": int(selected.get("size_bytes") or 0),
        }

    try:
        target = resolve_chat_target(
            selected,
            model_id=model_id,
            ollama_model=None,
            inference_backend=inference_backend,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    backend = target.get("inference_backend", inference_backend)
    if backend == BACKEND_OLLAMA:
        response = {
            "status": "ready",
            "backend": BACKEND_OLLAMA,
            "ollama_model": target.get("ollama_model"),
            "active_model": target.get("ollama_model"),
        }
        return {
            "ollama_only": True,
            "response": response,
            "size_bytes": int(selected.get("size_bytes") or 0),
        }

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
    if _active_local_model_would_change(pool, target_path=resolved_path, backend=payload.get("inference_backend")):
        pool.cancel_and_unload()
    if route == "mlx":
        pool.get_mlx(resolved_path)
    elif route == "torch":
        pool.get_torch(resolved_path)
    else:
        pool.get_llama(resolved_path, n_ctx=payload.get("n_ctx", 4096))


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
        active_path
        and pool.normalize_path(active_path) != pool.normalize_path(target_path)
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

    if body.defense is True and not settings.autodefense_enabled:
        raise HTTPException(400, "AutoDefense is not enabled on this server (SEISO_AUTODEFENSE_ENABLED)")

    job_id = orchestrator.create_job(user_id=user_id)
    payload = body.model_dump(exclude={"messages"})
    payload["user_id"] = user_id

    trusted_messages, _user_content = await build_trusted_messages(
        db,
        thread_id=body.thread_id,
        client_messages=body.messages,
        persist_user=bool(body.thread_id),
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
            if selected and selected.get("memory_load_blocked"):
                raise HTTPException(
                    400,
                    selected.get("memory_load_blocked_reason") or "Model exceeds available memory on this machine",
                )
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

            if target.get("inference_backend") == BACKEND_OLLAMA:
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

    if body.stream:
        can_stream_local = not body.tools and not body.provider_id
        use_defense = defense_enabled(settings, request_flag=body.defense)

        async def event_gen():
            if can_stream_local:
                parts: list[str] = []
                backend_label = payload.get("inference_backend") or "local"
                cancelled = False
                try:
                    orchestrator._emit_log(job_id, f"Streaming inference ({backend_label})")
                    stream_guard = StreamingOutputSanitizer()
                    async for token in orchestrator.stream_local(payload):
                        parts.append(token)
                        if not use_defense:
                            for safe_token in stream_guard.feed(token):
                                yield {"event": "token", "data": safe_token}
                    content = sanitize_llm_output("".join(parts))
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
                    if use_defense:
                        for token in chunk_sanitized_output(content):
                            yield {"event": "token", "data": token}
                    else:
                        for token in stream_guard.finish():
                            yield {"event": "token", "data": token}
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
                if job.result.get("defense"):
                    yield {"event": "defense", "data": json.dumps(job.result["defense"])}
            elif job and job.result.get("content"):
                content = sanitize_llm_output(job.result["content"])
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
    if job.result.get("content"):
        job.result["content"] = sanitize_llm_output(job.result["content"])
    if body.thread_id and job.result.get("content"):
        await db.add_message(body.thread_id, "assistant", job.result["content"])
    return {"job_id": job_id, **job.result}
