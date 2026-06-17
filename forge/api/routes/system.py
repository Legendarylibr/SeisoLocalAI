"""Local system hardware profile and live metrics — trustless, no telemetry."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from forge.security.auth import get_current_user_id
from forge.services.hardware import build_guidance, hardware_profile, hardware_summary, live_metrics
from seiso.models.loader import Backend, detect_backend

router = APIRouter(prefix="/system", tags=["system"])


class GuidanceRequest(BaseModel):
    goal: str = Field(default="chat", pattern="^(chat|train|compress|code|explore)$")


@router.get("/hardware")
async def get_hardware(
    user_id: Annotated[str, Depends(get_current_user_id)],
) -> dict[str, Any]:
    return hardware_profile()


@router.get("/metrics")
async def get_metrics(
    user_id: Annotated[str, Depends(get_current_user_id)],
) -> dict[str, Any]:
    return live_metrics()


@router.get("/guide")
async def get_guide(
    user_id: Annotated[str, Depends(get_current_user_id)],
    goal: str = Query("chat", pattern="^(chat|train|compress|inference|code|explore)$"),
) -> dict[str, Any]:
    profile = hardware_profile()
    backend = Backend(profile["backend"])
    steps = build_guidance(
        goal,
        backend=backend,
        gpus=profile["gpus"],
        ram_gb=profile["ram_gb"],
    )
    return {
        "goal": goal,
        "steps": [{"title": s.title, "detail": s.detail, "path": s.path} for s in steps],
        "hardware_summary": hardware_summary(profile),
        "local_only": True,
    }
