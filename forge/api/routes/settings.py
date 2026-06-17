"""Settings and health routes."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from forge.config import ForgeSettings, get_settings
from forge.security.auth import get_current_user_id
from seiso.models.loader import detect_backend

router = APIRouter(tags=["settings"])


class SettingsView(BaseModel):
    host: str
    port: int
    data_dir: str
    backend: str
    allow_remote: bool
    hf_configured: bool


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
    )
