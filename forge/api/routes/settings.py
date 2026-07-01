"""Settings and health routes."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from forge.config import ForgeSettings, get_settings
from forge.security.auth import get_current_user_id
from forge.services.hf_auth import (
    clear_user_hf_token,
    hf_auth_status,
    save_user_hf_token,
)
from forge.services.hf_connectivity import check_inference_runtime
from seiso.models.loader import detect_backend

router = APIRouter(tags=["settings"])


class SecurityPosture(BaseModel):
    """Server-side security capabilities exposed to the authenticated UI."""

    allow_tools: bool
    allow_code_exec: bool
    allow_openai_tools: bool
    allow_remote: bool
    bind_localhost: bool
    db_encrypted: bool
    rate_limit: int
    rate_limit_enabled: bool
    session_hours: int


class HfAuthView(BaseModel):
    cli_available: bool
    cli_binary: str | None
    cli_logged_in: bool
    token_configured: bool
    token_sources: list[str]
    user_token_saved: bool


class SettingsView(BaseModel):
    host: str
    port: int
    data_dir: str
    training_backend: str
    inference_backends: list[str]
    allow_remote: bool
    hf_configured: bool
    hf_auth: HfAuthView
    security: SecurityPosture


class HfTokenUpdate(BaseModel):
    token: str = Field(min_length=1, max_length=512)


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "seiso-forge"}


@router.get("/settings", response_model=SettingsView)
async def get_app_settings(
    user_id: Annotated[str, Depends(get_current_user_id)],
    settings: Annotated[ForgeSettings, Depends(get_settings)],
) -> SettingsView:
    from forge.services.hf_auth import load_user_hf_token

    auth = hf_auth_status(
        user_id=user_id,
        data_dir=settings.data_dir,
        encryption_key=settings.hf_token_encryption_key,
        settings_token=settings.hf_token or None,
    )
    user_saved = bool(
        load_user_hf_token(
            settings.data_dir, user_id, encryption_key=settings.hf_token_encryption_key
        )
    )
    runtime = check_inference_runtime()
    inference_backends: list[str] = []
    if runtime.llamacpp:
        inference_backends.append("llamacpp")
    if runtime.llamaswap:
        inference_backends.append("llamaswap")
    if runtime.mlx:
        inference_backends.append("mlx")
    if runtime.torch:
        inference_backends.append("torch")
    return SettingsView(
        host=settings.host,
        port=settings.port,
        data_dir=str(settings.data_dir),
        training_backend=detect_backend().value,
        inference_backends=inference_backends,
        allow_remote=settings.allow_remote,
        hf_configured=auth.token_configured,
        hf_auth=HfAuthView(
            cli_available=auth.cli_available,
            cli_binary=auth.cli_binary,
            cli_logged_in=auth.cli_logged_in,
            token_configured=auth.token_configured,
            token_sources=auth.token_sources,
            user_token_saved=user_saved,
        ),
        security=SecurityPosture(
            allow_tools=settings.allow_tools,
            allow_code_exec=settings.allow_code_exec,
            allow_openai_tools=settings.allow_openai_tools,
            allow_remote=settings.allow_remote,
            bind_localhost=not settings.allow_remote,
            db_encrypted=True,
            rate_limit=settings.rate_limit,
            rate_limit_enabled=settings.rate_limit_enabled,
            session_hours=settings.session_hours,
        ),
    )


@router.put("/settings/hf-token")
async def save_hf_token(
    body: HfTokenUpdate,
    user_id: Annotated[str, Depends(get_current_user_id)],
    settings: Annotated[ForgeSettings, Depends(get_settings)],
) -> dict[str, str]:
    from forge.services.hf_auth import _normalize_token
    from forge.services.hf_connectivity import probe_hf_hub

    token = _normalize_token(body.token)
    if not token:
        raise HTTPException(
            status_code=400, detail="Invalid Hugging Face token format."
        )

    result = probe_hf_hub(token=token)
    if not result.reachable:
        raise HTTPException(
            status_code=400,
            detail=result.error or "Cannot reach Hugging Face Hub to validate token.",
        )
    if result.token_invalid:
        raise HTTPException(
            status_code=400,
            detail="Hugging Face rejected this token — generate a new one at huggingface.co/settings/tokens.",
        )

    save_user_hf_token(
        settings.data_dir,
        user_id,
        token,
        encryption_key=settings.hf_token_encryption_key,
    )
    return {"status": "saved"}


@router.delete("/settings/hf-token")
async def delete_hf_token(
    user_id: Annotated[str, Depends(get_current_user_id)],
    settings: Annotated[ForgeSettings, Depends(get_settings)],
) -> dict[str, str]:
    clear_user_hf_token(settings.data_dir, user_id)
    return {"status": "cleared"}


@router.get("/settings/hf-status")
async def hf_hub_status(
    user_id: Annotated[str, Depends(get_current_user_id)],
    settings: Annotated[ForgeSettings, Depends(get_settings)],
) -> dict:
    """Probe Hugging Face Hub connectivity, auth, transfer stack, and inference deps."""
    from forge.services.hf_connectivity import build_hf_status

    return build_hf_status(
        user_id=user_id,
        data_dir=settings.data_dir,
        encryption_key=settings.hf_token_encryption_key,
        settings_token=settings.hf_token or None,
        probe=True,
    )
