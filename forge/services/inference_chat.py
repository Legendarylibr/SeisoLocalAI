"""Forge inference chat helpers — model resolution, memory checks, preload."""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import HTTPException

from forge.config import ForgeSettings
from forge.db.store import Database
from forge.services.inference_models import (
    get_inference_option,
    list_inference_options,
    resolve_chat_target,
)
from forge.services.model_router_client import ROUTER_MODEL_ID
from forge.services.models import resolve_model_path
from seiso.inference.backends import BACKEND_LLAMACPP, BACKEND_ROUTER, BACKEND_TORCH
from seiso.inference.runner import LocalInferenceRunner


def assert_model_fits_for_load(
    path: str,
    *,
    mode: str = "chat",
    backend: str | None = None,
) -> None:
    from seiso.memory.protection import assess_path_memory_fit_for_load

    fit = assess_path_memory_fit_for_load(path, mode=mode, backend=backend)
    if fit.get("memory_load_blocked"):
        raise HTTPException(
            400,
            fit.get("memory_load_blocked_reason")
            or "Model exceeds available memory on this machine",
        )


async def resolve_preload_context(
    db: Database,
    user_id: str,
    settings: ForgeSettings,
    model_id: str,
    inference_backend: str,
    *,
    max_tokens: int = 2048,
    n_ctx: int | None = None,
) -> dict[str, Any]:
    selected = await get_inference_option(db, user_id, model_id)
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

    assert_model_fits_for_load(path, mode="chat", backend=backend)

    payload = {
        "model_path": path,
        "model_format": target.get("model_format") or selected.get("format"),
        "inference_backend": backend,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": max_tokens,
    }
    if n_ctx is not None:
        payload["n_ctx"] = n_ctx
    return {
        "payload": payload,
        "backend": backend,
        "model_name": selected.get("name") or model_id,
        "size_bytes": int(selected.get("size_bytes") or 0),
    }


async def release_active_local_model(runner: LocalInferenceRunner) -> None:
    if not runner.pool.active_key:
        return
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, runner.pool.cancel_and_unload)


async def resolve_inventory_model_path(
    db: Database,
    user_id: str,
    settings: ForgeSettings,
    *,
    model_id: str,
    model_path: str | None,
    inference_backend: str,
    model_router_enabled: bool,
) -> dict[str, Any]:
    """Resolve model_id to payload fields (path, backend, router flags)."""
    selected = await get_inference_option(db, user_id, model_id)
    if selected is None and model_router_enabled and model_id == ROUTER_MODEL_ID:
        selected = next(
            (
                o
                for o in await list_inference_options(
                    db,
                    user_id,
                    model_router_enabled=True,
                )
                if o["id"] == model_id
            ),
            None,
        )
    try:
        target = resolve_chat_target(
            selected,
            model_id=model_id,
            inference_backend=inference_backend,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    updates: dict[str, Any] = {
        "inference_backend": target.get("inference_backend", inference_backend),
        "model_format": target.get("model_format"),
    }

    if target.get("inference_backend") == BACKEND_ROUTER:
        if not model_router_enabled:
            raise HTTPException(
                400,
                "Smart Router is not enabled (set SEISO_MODEL_ROUTER_ENABLED=true)",
            )
        updates["use_model_router"] = True
        return updates

    path = target.get("model_path")
    if not path and model_id:
        path = await resolve_model_path(
            db,
            user_id,
            model_id=model_id,
            model_path=model_path,
            data_dir=settings.data_dir,
        )
    if not path:
        raise HTTPException(400, "Select a model from inventory or provide model_path")

    assert_model_fits_for_load(
        path,
        mode="chat",
        backend=updates.get("inference_backend"),
    )
    updates["model_path"] = path
    return updates


async def resolve_explicit_model_path(
    db: Database,
    user_id: str,
    settings: ForgeSettings,
    *,
    model_path: str,
    inference_backend: str,
) -> dict[str, Any]:
    path = await resolve_model_path(
        db,
        user_id,
        model_id=None,
        model_path=model_path,
        data_dir=settings.data_dir,
    )
    if not path:
        raise HTTPException(400, "Invalid model_path")
    assert_model_fits_for_load(path, mode="chat", backend=inference_backend)
    return {
        "model_path": path,
        "inference_backend": inference_backend,
    }


async def resolve_draft_model(
    db: Database,
    user_id: str,
    settings: ForgeSettings,
    *,
    draft_model_id: str | None,
    draft_model_path: str | None,
) -> dict[str, Any]:
    if draft_model_id and draft_model_path:
        raise HTTPException(403, "Provide draft_model_id or draft_model_path, not both")

    if draft_model_path:
        draft_path = await resolve_model_path(
            db,
            user_id,
            model_id=None,
            model_path=draft_model_path,
            data_dir=settings.data_dir,
        )
    elif draft_model_id:
        draft_selected = await get_inference_option(db, user_id, draft_model_id)
        if not draft_selected:
            raise HTTPException(404, "Draft model not found")
        draft_path = draft_selected.get("path")
        if not draft_path:
            raise HTTPException(
                400, "Draft model must be a local safetensors/checkpoint path"
            )
    else:
        return {}

    if not draft_path:
        raise HTTPException(400, "Invalid draft model path")

    from seiso.inference.backends import is_dflash_draft

    draft_backend = BACKEND_LLAMACPP if is_dflash_draft(draft_path) else BACKEND_TORCH
    assert_model_fits_for_load(draft_path, mode="chat", backend=draft_backend)
    return {
        "draft_model_path": draft_path,
        "inference_backend": BACKEND_TORCH,
    }
