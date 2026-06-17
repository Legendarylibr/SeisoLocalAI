"""External LLM provider CRUD."""

from __future__ import annotations

import json
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from forge.api.deps import get_db
from forge.db.store import Database
from forge.providers.router import mask_config
from forge.security.audit import audit_event
from forge.security.auth import get_current_user_id
from seiso.security import SecurityError
from forge.security.url_policy import validate_provider_base_url

router = APIRouter(prefix="/providers", tags=["providers"])


class ProviderCreate(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    provider_type: str = Field(description="openai | anthropic | ollama | vllm")
    config: dict[str, Any] = Field(default_factory=dict)


@router.get("")
async def list_providers(
    user_id: Annotated[str, Depends(get_current_user_id)],
    db: Annotated[Database, Depends(get_db)],
) -> list[dict]:
    rows = await db.list_providers(user_id)
    return [
        {
            **{k: r[k] for k in ("id", "name", "provider_type", "created_at")},
            "config": mask_config(json.loads(r["config_json"])),
        }
        for r in rows
    ]


@router.post("", status_code=201)
async def create_provider(
    body: ProviderCreate,
    user_id: Annotated[str, Depends(get_current_user_id)],
    db: Annotated[Database, Depends(get_db)],
) -> dict:
    allowed = {"openai", "anthropic", "ollama", "vllm"}
    ptype = body.provider_type.lower()
    if ptype not in allowed:
        raise HTTPException(400, f"provider_type must be one of {allowed}")
    config = dict(body.config)
    if "base_url" in config and config["base_url"]:
        try:
            config["base_url"] = validate_provider_base_url(config["base_url"], provider_type=ptype)
        except SecurityError as exc:
            raise HTTPException(400, str(exc)) from exc
    row = await db.create_provider(user_id, body.name, ptype, config)
    row["config"] = mask_config(config)
    audit_event("provider_create", user_id=user_id, provider_id=row["id"], provider_type=ptype)
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
