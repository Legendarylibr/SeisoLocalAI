"""Forge inference chat helpers — single resolve/sanitize path for chat, preload, OpenAI, context."""

from __future__ import annotations

from pathlib import Path
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
from seiso.inference.backends import (
    BACKEND_LLAMACPP,
    BACKEND_LLAMASWAP,
    BACKEND_MLX,
    BACKEND_ROUTER,
    BACKEND_TORCH,
)


def assert_model_fits_for_load(
    path: str,
    *,
    mode: str = "chat",
    backend: str | None = None,
) -> None:
    from seiso.memory.protection import assess_path_memory_fit_for_load

    fit = assess_path_memory_fit_for_load(path, mode=mode, backend=backend)
    if mode != "chat" and fit.get("memory_load_blocked"):
        raise HTTPException(
            400,
            fit.get("memory_load_blocked_reason")
            or "Model exceeds available memory on this machine",
        )


def assert_backend_runtime_available(backend: str) -> None:
    from forge.services.hf_connectivity import check_inference_runtime

    runtime = check_inference_runtime()
    available = {
        BACKEND_LLAMACPP: runtime.llamacpp,
        BACKEND_LLAMASWAP: runtime.llamaswap,
        BACKEND_MLX: runtime.mlx,
        BACKEND_TORCH: runtime.torch,
    }
    if not available.get(backend, False):
        raise HTTPException(400, f"Inference backend {backend!r} is not available")


def _sanitize_chat_fields(
    *,
    model_path: str | None,
    model_format: str | None,
    max_tokens: int | None,
    n_ctx: int | None,
    messages: list[dict[str, Any]] | None,
    inference_backend: str | None = None,
) -> dict[str, Any]:
    from seiso.memory.protection import sanitize_inference_payload

    isolated = False
    if model_path:
        try:
            from seiso.inference.backends import (
                BACKEND_LLAMASWAP,
                resolve_local_backend,
            )

            isolated = (
                resolve_local_backend(
                    model_path=model_path,
                    model_format=model_format,
                    requested=inference_backend,
                )
                == BACKEND_LLAMASWAP
            )
        except Exception:
            isolated = False

    payload: dict[str, Any] = {
        "model_path": model_path,
        "model_format": model_format,
        "messages": messages or [],
        "max_tokens": max_tokens if max_tokens is not None else 2048,
    }
    if n_ctx is not None:
        payload["n_ctx"] = n_ctx
    sanitized = sanitize_inference_payload(payload, isolated=isolated)
    out: dict[str, Any] = {"max_tokens": sanitized["max_tokens"]}
    if "n_ctx" in sanitized:
        out["n_ctx"] = sanitized["n_ctx"]
    return out


async def _lookup_inference_option(
    db: Database,
    user_id: str,
    model_id: str,
    *,
    model_router_enabled: bool,
) -> dict[str, Any] | None:
    selected = await get_inference_option(db, user_id, model_id)
    if selected is not None:
        return selected
    if model_router_enabled and model_id == ROUTER_MODEL_ID:
        return next(
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
    return None


async def prepare_local_chat_target(
    db: Database,
    user_id: str,
    settings: ForgeSettings,
    *,
    model_id: str | None = None,
    model_path: str | None = None,
    inference_backend: str = "auto",
    model_router_enabled: bool = False,
    max_tokens: int | None = None,
    n_ctx: int | None = None,
    messages: list[dict[str, Any]] | None = None,
    check_memory: bool = True,
    sanitize: bool = False,
) -> dict[str, Any]:
    """Canonical resolve path for inventory/explicit local chat models.

    Used by Forge chat, preload, context, and OpenAI-compatible routes.
    """
    if model_id and model_path:
        raise HTTPException(403, "Provide model_id or model_path, not both")

    if model_path and not model_id:
        path = await resolve_model_path(
            db,
            user_id,
            model_id=None,
            model_path=model_path,
            data_dir=settings.data_dir,
        )
        if not path:
            raise HTTPException(400, "Invalid model_path")
        backend = (inference_backend or "auto").lower()
        from seiso.inference.backends import resolve_local_backend

        try:
            backend = resolve_local_backend(
                model_path=path,
                model_format=None,
                requested=backend,
            )
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(400, str(exc)) from exc
        assert_backend_runtime_available(backend)
        if check_memory:
            assert_model_fits_for_load(path, mode="chat", backend=backend)
        from seiso.inference.ollama_registry import metadata_for_model_path

        path_meta = metadata_for_model_path(path)
        updates: dict[str, Any] = {
            "model_path": path,
            "inference_backend": backend,
        }
        if path_meta:
            updates["model_metadata"] = path_meta
        if sanitize:
            updates.update(
                _sanitize_chat_fields(
                    model_path=path,
                    model_format=None,
                    max_tokens=max_tokens,
                    n_ctx=n_ctx,
                    messages=messages,
                    inference_backend=backend,
                )
            )
        return updates

    if not model_id:
        raise HTTPException(400, "Select a model from inventory or provide model_path")

    selected = await _lookup_inference_option(
        db,
        user_id,
        model_id,
        model_router_enabled=model_router_enabled,
    )
    if selected is None:
        raise HTTPException(404, "Model not found in inventory")

    if not selected.get("selectable", True) and selected.get("kind") != "router":
        raise HTTPException(
            400,
            selected.get("hardware_note")
            or selected.get("status_note")
            or "Download incomplete or model path missing. Re-download from Hub.",
        )

    try:
        target = resolve_chat_target(
            selected,
            model_id=model_id,
            inference_backend=inference_backend,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    updates = {
        "inference_backend": target.get("inference_backend", inference_backend),
        "model_format": target.get("model_format") or selected.get("format"),
        "model_name": selected.get("name") or model_id,
        "size_bytes": int(selected.get("size_bytes") or 0),
        "context_ceiling": selected.get("context_ceiling"),
        "architecture": selected.get("architecture"),
        "is_moe": selected.get("is_moe"),
        "uses_swa": selected.get("uses_swa"),
    }
    if selected.get("metadata"):
        updates["model_metadata"] = selected["metadata"]

    if target.get("inference_backend") == BACKEND_ROUTER:
        if not model_router_enabled:
            raise HTTPException(
                400,
                "Smart Router is not enabled (set SEISO_MODEL_ROUTER_ENABLED=true)",
            )
        updates["use_model_router"] = True
        updates["model_path"] = None
        return updates

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
        raise HTTPException(
            400,
            "Download incomplete or model path missing. Re-download from Hub.",
        )

    if check_memory:
        assert_model_fits_for_load(
            path,
            mode="chat",
            backend=updates.get("inference_backend"),
        )
    updates["model_path"] = path

    if sanitize:
        updates.update(
            _sanitize_chat_fields(
                model_path=path,
                model_format=updates.get("model_format"),
                max_tokens=max_tokens,
                n_ctx=n_ctx,
                messages=messages,
                inference_backend=updates.get("inference_backend"),
            )
        )
    return updates


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
    target = await prepare_local_chat_target(
        db,
        user_id,
        settings,
        model_id=model_id,
        inference_backend=inference_backend,
        model_router_enabled=False,
        max_tokens=max_tokens,
        n_ctx=n_ctx,
        messages=[{"role": "user", "content": "ping"}],
        check_memory=True,
        sanitize=True,
    )
    if target.get("use_model_router"):
        raise HTTPException(400, "Smart Router does not support model preload")

    path = target.get("model_path")
    if not path:
        raise HTTPException(400, "Model path not found")

    payload: dict[str, Any] = {
        "model_path": path,
        "model_format": target.get("model_format"),
        "inference_backend": target["inference_backend"],
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": target.get("max_tokens", max_tokens),
    }
    if target.get("model_metadata"):
        payload["model_metadata"] = target["model_metadata"]
    if target.get("n_ctx") is not None:
        payload["n_ctx"] = target["n_ctx"]
    elif n_ctx is not None:
        payload["n_ctx"] = n_ctx

    return {
        "payload": payload,
        "backend": target["inference_backend"],
        "model_name": target.get("model_name") or model_id,
        "size_bytes": int(target.get("size_bytes") or 0),
    }


async def resolve_inventory_model_path(
    db: Database,
    user_id: str,
    settings: ForgeSettings,
    *,
    model_id: str,
    model_path: str | None,
    inference_backend: str,
    model_router_enabled: bool,
    max_tokens: int | None = None,
    n_ctx: int | None = None,
    messages: list[dict[str, Any]] | None = None,
    sanitize: bool = False,
) -> dict[str, Any]:
    """Resolve model_id to payload fields (path, backend, router flags)."""
    return await prepare_local_chat_target(
        db,
        user_id,
        settings,
        model_id=model_id,
        model_path=model_path,
        inference_backend=inference_backend,
        model_router_enabled=model_router_enabled,
        max_tokens=max_tokens,
        n_ctx=n_ctx,
        messages=messages,
        check_memory=True,
        sanitize=sanitize,
    )


async def resolve_explicit_model_path(
    db: Database,
    user_id: str,
    settings: ForgeSettings,
    *,
    model_path: str,
    inference_backend: str,
    max_tokens: int | None = None,
    n_ctx: int | None = None,
    messages: list[dict[str, Any]] | None = None,
    sanitize: bool = False,
) -> dict[str, Any]:
    return await prepare_local_chat_target(
        db,
        user_id,
        settings,
        model_path=model_path,
        inference_backend=inference_backend,
        max_tokens=max_tokens,
        n_ctx=n_ctx,
        messages=messages,
        check_memory=True,
        sanitize=sanitize,
    )


def _vocab_size_from_path(model_path: str) -> int | None:
    root = Path(model_path)
    if root.is_file():
        root = root.parent
    config = root / "config.json"
    if not config.is_file():
        return None
    try:
        from seiso.io.jsonl import read_json_file

        data = read_json_file(config, default=None)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    for key in ("vocab_size", "padded_vocab_size"):
        value = data.get(key)
        if isinstance(value, int) and value > 0:
            return value
    return None


def _assert_draft_compatible(target_path: str | None, draft_path: str) -> None:
    from seiso.inference.backends import (
        gguf_architecture,
        is_dflash_draft,
    )

    if is_dflash_draft(draft_path):
        draft_arch = (gguf_architecture(draft_path) or "").lower()
        if target_path:
            target_arch = (gguf_architecture(target_path) or "").lower()
            if target_arch and draft_arch and "dflash" not in draft_arch:
                target_family = target_arch.split("-", 1)[0]
                draft_family = draft_arch.split("-", 1)[0]
                if target_family and draft_family and target_family != draft_family:
                    raise HTTPException(
                        400,
                        f"Draft architecture {draft_arch!r} is incompatible with target {target_arch!r}",
                    )
        return

    if not target_path:
        return

    target_vocab = _vocab_size_from_path(target_path)
    draft_vocab = _vocab_size_from_path(draft_path)
    if target_vocab is not None and draft_vocab is not None and target_vocab != draft_vocab:
        raise HTTPException(
            400,
            f"Draft/target tokenizers appear incompatible: vocab_size target={target_vocab} draft={draft_vocab}",
        )


async def resolve_draft_model(
    db: Database,
    user_id: str,
    settings: ForgeSettings,
    *,
    draft_model_id: str | None,
    draft_model_path: str | None,
    target_model_path: str | None = None,
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
        compatibility_checked = False
    elif draft_model_id:
        draft_selected = await get_inference_option(db, user_id, draft_model_id)
        if not draft_selected:
            raise HTTPException(404, "Draft model not found")
        if not draft_selected.get("selectable", True):
            raise HTTPException(
                400,
                draft_selected.get("hardware_note")
                or "Draft model download is incomplete",
            )
        selected_path = str(draft_selected.get("path") or "")
        if not selected_path:
            raise HTTPException(
                400, "Draft model must be a local safetensors/checkpoint path"
            )
        if target_model_path:
            _assert_draft_compatible(target_model_path, selected_path)
        compatibility_checked = bool(target_model_path)
        draft_path = await resolve_model_path(
            db,
            user_id,
            model_id=draft_model_id,
            model_path=None,
            data_dir=settings.data_dir,
        )
        if not draft_path:
            raise HTTPException(
                400, "Draft model must be a local safetensors/checkpoint path"
            )
    else:
        return {}

    if not draft_path:
        raise HTTPException(400, "Invalid draft model path")

    from seiso.inference.backends import (
        _native_linux_requires_isolated_gguf,
        is_dflash_draft,
    )

    if not compatibility_checked:
        _assert_draft_compatible(target_model_path, draft_path)

    is_dflash = is_dflash_draft(draft_path)
    if is_dflash and _native_linux_requires_isolated_gguf():
        raise HTTPException(
            400,
            "dFlash speculative decoding uses an in-process llama.cpp GGUF draft, "
            "which is blocked on native Linux NVIDIA. Disable speculative decoding "
            "or set SEISO_LLAMA_ALLOW_INPROCESS_NATIVE_LINUX=1 to explicitly accept "
            "the in-process llama.cpp risk.",
        )

    draft_backend = BACKEND_LLAMACPP if is_dflash else BACKEND_TORCH
    assert_backend_runtime_available(draft_backend)
    assert_model_fits_for_load(draft_path, mode="chat", backend=draft_backend)
    return {
        "draft_model_path": draft_path,
        "inference_backend": BACKEND_TORCH,
    }
