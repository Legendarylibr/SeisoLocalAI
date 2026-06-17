"""AutoDefense health check and calibration routes."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from forge.config import ForgeSettings, get_settings
from forge.security.auth import get_current_user_id
from forge.security.autodefense import analyze, check_health, defense_enabled

router = APIRouter(prefix="/autodefense", tags=["autodefense"])


class AnalyzeRequest(BaseModel):
    user_input: str = Field(..., max_length=50_000)
    model_output: str | None = Field(default=None, max_length=100_000)
    session_id: str | None = None


@router.get("/health")
async def autodefense_health(
    user_id: Annotated[str, Depends(get_current_user_id)],
    settings: Annotated[ForgeSettings, Depends(get_settings)],
) -> dict[str, Any]:
    """Probe whether AutoDefense is enabled and reachable."""
    health = await check_health(settings)
    return {
        **health,
        "configured": settings.autodefense_enabled,
        "url": settings.autodefense_url if settings.autodefense_enabled else None,
        "fail_open": settings.autodefense_fail_open,
    }


@router.post("/analyze")
async def autodefense_analyze(
    body: AnalyzeRequest,
    user_id: Annotated[str, Depends(get_current_user_id)],
    settings: Annotated[ForgeSettings, Depends(get_settings)],
) -> dict[str, Any]:
    """Run AutoDefense analysis without invoking an LLM (calibration / prompt-injection testing)."""
    if not defense_enabled(settings):
        raise HTTPException(503, "AutoDefense is disabled (set SEISO_AUTODEFENSE_ENABLED=true)")

    result = await analyze(
        body.user_input,
        model_output=body.model_output,
        session_id=body.session_id,
        metadata={"source": "seiso_calibration", "user_id": user_id},
        settings=settings,
    )
    return {
        **result.to_dict(),
        "sanitized_input": result.sanitized_input,
        "sanitized_output": result.sanitized_output,
    }
