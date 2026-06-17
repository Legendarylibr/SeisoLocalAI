"""Settings and health routes."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from forge.config import ForgeSettings, get_settings
from forge.security.auth import get_current_user_id
from seiso.models.loader import detect_backend

router = APIRouter(tags=["settings"])


class SecurityPosture(BaseModel):
    """Server-side security capabilities exposed to the authenticated UI."""

    allow_tools: bool
    allow_code_exec: bool
    allow_openai_tools: bool
    allow_remote: bool
    autodefense_enabled: bool
    bind_localhost: bool
    db_encrypted: bool
    rate_limit: int
    session_hours: int


class SettingsView(BaseModel):
    host: str
    port: int
    data_dir: str
    backend: str
    allow_remote: bool
    hf_configured: bool
    autodefense_enabled: bool
    autodefense_configured: bool
    security: SecurityPosture


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "seiso-forge"}


@router.get("/settings", response_model=SettingsView)
async def get_app_settings(
    user_id: Annotated[str, Depends(get_current_user_id)],
    settings: Annotated[ForgeSettings, Depends(get_settings)],
) -> SettingsView:
    return SettingsView(
        host=settings.host,
        port=settings.port,
        data_dir=str(settings.data_dir),
        backend=detect_backend().value,
        allow_remote=settings.allow_remote,
        hf_configured=bool(settings.hf_token),
        autodefense_enabled=settings.autodefense_enabled,
        autodefense_configured=settings.autodefense_enabled,
        security=SecurityPosture(
            allow_tools=settings.allow_tools,
            allow_code_exec=settings.allow_code_exec,
            allow_openai_tools=settings.allow_openai_tools,
            allow_remote=settings.allow_remote,
            autodefense_enabled=settings.autodefense_enabled,
            bind_localhost=not settings.allow_remote,
            db_encrypted=True,
            rate_limit=settings.rate_limit,
            session_hours=settings.session_hours,
        ),
    )
