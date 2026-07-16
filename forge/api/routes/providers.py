"""External chat-server CRUD — local_chat + optional remote_chat + managed multi-GPU."""

from __future__ import annotations

import json
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from forge.api.deps import get_db
from forge.config import ForgeSettings, get_settings
from forge.db.store import Database
from forge.providers.router import (
    PROVIDER_LOCAL_CHAT,
    PROVIDER_REMOTE_CHAT,
    allowed_chat_provider_types,
    cloud_multigpu_enabled,
    is_chat_provider_type,
    mask_config,
    normalize_provider_type,
    normalize_remote_chat_config,
    preferred_chat_provider_types,
)
from forge.security.audit import audit_event
from forge.security.auth import get_current_user_id
from forge.security.url_policy import validate_provider_base_url
from forge.services import managed_vllm as managed_vllm_svc
from forge.services.memory_release import prepare_for_gpu_task, release_after_task
from seiso.inference.managed_vllm import managed_vllm_enabled
from seiso.security import SecurityError

router = APIRouter(prefix="/providers", tags=["providers"])

_REMOVED_FRONTIER_TYPES = frozenset({"openai", "anthropic"})


class ProviderCreate(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    provider_type: str = Field(
        description=(
            "local_chat (loopback multi-GPU/chat server) or "
            "remote_chat (remote HTTPS multi-GPU, opt-in). "
            "Aliases: vllm → local_chat, vllm_cloud → remote_chat."
        )
    )
    config: dict[str, Any] = Field(default_factory=dict)


class ManagedVllmStartRequest(BaseModel):
    model: str = Field(min_length=1, max_length=512)
    tensor_parallel_size: int | None = Field(default=None, ge=1, le=256)
    host: str = Field(default="127.0.0.1", max_length=64)
    port: int = Field(default=8000, ge=1, le=65535)
    cuda_visible_devices: str | None = Field(default=None, max_length=128)
    max_model_len: int | None = Field(default=None, ge=256, le=1_048_576)
    gpu_memory_utilization: float | None = Field(default=None, ge=0.1, le=1.0)
    wait_ready: bool = True


class ManagedVllmPreviewRequest(BaseModel):
    model: str = Field(min_length=1, max_length=512)
    tensor_parallel_size: int | None = Field(default=None, ge=1, le=256)
    host: str = Field(default="127.0.0.1", max_length=64)
    port: int = Field(default=8000, ge=1, le=65535)
    cuda_visible_devices: str | None = Field(default=None, max_length=128)
    max_model_len: int | None = Field(default=None, ge=256, le=1_048_576)
    gpu_memory_utilization: float | None = Field(default=None, ge=0.1, le=1.0)


def _chat_provider_rows(rows: list[dict]) -> list[dict]:
    allowed = allowed_chat_provider_types()
    return [
        {
            **{k: r[k] for k in ("id", "name", "provider_type", "created_at")},
            "config": mask_config(json.loads(r["config_json"])),
        }
        for r in rows
        if r["provider_type"].lower() in allowed
    ]


@router.get("")
async def list_providers(
    user_id: Annotated[str, Depends(get_current_user_id)],
    db: Annotated[Database, Depends(get_db)],
) -> list[dict]:
    """List chat providers only — never cloud_gpu credential rows."""
    rows = await db.list_providers(user_id)
    return _chat_provider_rows(rows)


@router.post("", status_code=201)
async def create_provider(
    body: ProviderCreate,
    user_id: Annotated[str, Depends(get_current_user_id)],
    db: Annotated[Database, Depends(get_db)],
) -> dict:
    raw_type = body.provider_type.lower()
    if raw_type in _REMOVED_FRONTIER_TYPES:
        raise HTTPException(400, "Frontier cloud providers are not supported")
    if not is_chat_provider_type(raw_type):
        preferred = preferred_chat_provider_types()
        raise HTTPException(
            400,
            f"provider_type must be one of {preferred} "
            f"(also accepts aliases {sorted(allowed_chat_provider_types() - set(preferred))})",
        )
    # Store canonical names so UI/agents stay vendor-neutral.
    ptype = normalize_provider_type(raw_type)
    if ptype == PROVIDER_REMOTE_CHAT and not cloud_multigpu_enabled():
        raise HTTPException(
            400,
            "Remote multi-GPU chat servers are disabled. "
            "Set SEISO_ALLOW_CLOUD_MULTIGPU=true to enable this optional path.",
        )
    config = dict(body.config)
    if ptype == PROVIDER_REMOTE_CHAT:
        try:
            config = normalize_remote_chat_config(config)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        if not config.get("base_url"):
            raise HTTPException(400, "remote_chat requires config.base_url (HTTPS)")
        if not config.get("model"):
            raise HTTPException(400, "remote_chat requires config.model")
    if "base_url" in config and config["base_url"]:
        try:
            config["base_url"] = validate_provider_base_url(
                config["base_url"], provider_type=ptype
            )
        except SecurityError as exc:
            raise HTTPException(400, str(exc)) from exc
    if ptype == PROVIDER_LOCAL_CHAT and not config.get("base_url"):
        config["base_url"] = validate_provider_base_url("", provider_type=ptype)
        config.setdefault("deployment_kind", "multi_gpu_local")
    row = await db.create_provider(user_id, body.name, ptype, config)
    row["config"] = mask_config(config)
    audit_event(
        "provider_create", user_id=user_id, provider_id=row["id"], provider_type=ptype
    )
    return row


@router.delete("/{provider_id}")
async def delete_provider(
    provider_id: str,
    user_id: Annotated[str, Depends(get_current_user_id)],
    db: Annotated[Database, Depends(get_db)],
) -> dict:
    ok = await db.delete_provider(provider_id, user_id)
    if not ok:
        raise HTTPException(404, "Provider not found")
    audit_event("provider_delete", user_id=user_id, provider_id=provider_id)
    return {"deleted": True}


# --- Optional managed multi-GPU vLLM (local) ---


@router.get("/managed-vllm/status")
async def managed_vllm_status(
    user_id: Annotated[str, Depends(get_current_user_id)],
) -> dict:
    status = managed_vllm_svc.get_status()
    status["feature_enabled"] = managed_vllm_enabled()
    status["cloud_multigpu_enabled"] = cloud_multigpu_enabled()
    return status


@router.post("/managed-vllm/preview")
async def managed_vllm_preview(
    body: ManagedVllmPreviewRequest,
    user_id: Annotated[str, Depends(get_current_user_id)],
) -> dict:
    if not managed_vllm_enabled():
        raise HTTPException(
            400,
            "Managed multi-GPU vLLM is disabled. "
            "Set SEISO_MANAGED_VLLM_ENABLED=true for this optional path.",
        )
    try:
        return managed_vllm_svc.launch_preview(
            model=body.model,
            tensor_parallel_size=body.tensor_parallel_size,
            host=body.host,
            port=body.port,
            cuda_visible_devices=body.cuda_visible_devices,
            max_model_len=body.max_model_len,
            gpu_memory_utilization=body.gpu_memory_utilization,
        )
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/managed-vllm/start")
async def managed_vllm_start(
    body: ManagedVllmStartRequest,
    user_id: Annotated[str, Depends(get_current_user_id)],
    db: Annotated[Database, Depends(get_db)],
    settings: Annotated[ForgeSettings, Depends(get_settings)],
) -> dict:
    """Start local multi-GPU vLLM (optional). Registers a chat provider for Forge + Compat API."""
    if not managed_vllm_enabled():
        raise HTTPException(
            400,
            "Managed multi-GPU vLLM is disabled. "
            "Set SEISO_MANAGED_VLLM_ENABLED=true for this optional path.",
        )

    # Free local pool VRAM first; managed vLLM then owns the GPUs.
    prep = prepare_for_gpu_task(task="inference", user_id=user_id)
    resource_token = prep.get("resource_token")
    try:
        try:
            status = managed_vllm_svc.start_managed_vllm(
                model=body.model,
                data_dir=settings.data_dir,
                host=body.host,
                port=body.port,
                tensor_parallel_size=body.tensor_parallel_size,
                cuda_visible_devices=body.cuda_visible_devices,
                max_model_len=body.max_model_len,
                gpu_memory_utilization=body.gpu_memory_utilization,
                wait_ready=body.wait_ready,
            )
        except Exception as exc:
            raise HTTPException(400, str(exc)) from exc
        provider = await managed_vllm_svc.ensure_managed_provider_row(
            db, user_id, status
        )
    finally:
        # Drop short-lived task reservation; keep the managed server process.
        release_after_task(
            reason="managed_vllm_start",
            resource_token=str(resource_token) if resource_token else None,
        )
    audit_event(
        "managed_vllm_start",
        user_id=user_id,
        model=body.model,
        tensor_parallel_size=status.get("tensor_parallel_size"),
        provider_id=(provider or {}).get("id"),
    )
    return {
        "status": status,
        "provider": (
            {
                "id": provider["id"],
                "name": provider["name"],
                "provider_type": provider["provider_type"],
                "config": mask_config(provider.get("config") or {}),
            }
            if provider
            else None
        ),
        "compat": {
            "base_url": "http://127.0.0.1:8765/v1",
            "model_ids": (
                [f"provider:{provider['id']}"]
                + (
                    [status["model"]]
                    if status.get("model")
                    else []
                )
                if provider
                else []
            ),
            "note": (
                "Point external agents at Forge Compat API /v1 with the provider model id "
                "(or the upstream model name when listed). Local inventory models are unchanged."
            ),
        },
    }


@router.post("/managed-vllm/stop")
async def managed_vllm_stop(
    user_id: Annotated[str, Depends(get_current_user_id)],
    db: Annotated[Database, Depends(get_db)],
    settings: Annotated[ForgeSettings, Depends(get_settings)],
) -> dict:
    result = managed_vllm_svc.stop_managed_vllm(data_dir=settings.data_dir)
    removed = await managed_vllm_svc.remove_managed_provider_rows(db, user_id)
    audit_event("managed_vllm_stop", user_id=user_id, removed_providers=removed)
    return {**result, "removed_providers": removed}
