"""Training access surfaces: frontend (Forge UI) vs generic agent.

Mesh / multi-node coordination is Buzz-agent-only. Local single-node and
local multi-GPU DDP remain available on both surfaces.
"""

from __future__ import annotations

import os
from enum import Enum


class TrainingSurface(str, Enum):
    """Where a training request originated."""

    FRONTEND = "frontend"
    AGENT = "agent"


def _truthy(raw: str | None) -> bool:
    return (raw or "").strip().lower() in {"1", "true", "yes", "on"}


def buzz_agent_present() -> bool:
    """True when a Buzz agent identity is configured (presence only; never log values).

    ``BUZZ_PRIVATE_KEY`` is the agent signing identity used by buzz-cli.
    ``BUZZ_AUTH_TAG`` is set by managed Buzz Desktop agent sessions.
    """
    return bool(
        (os.environ.get("BUZZ_PRIVATE_KEY") or "").strip()
        or (os.environ.get("BUZZ_AUTH_TAG") or "").strip()
    )


def agent_context_present() -> bool:
    """True for any agent harness (Buzz or generic)."""
    if _truthy(os.environ.get("SEISO_AGENT")):
        return True
    surface = (os.environ.get("SEISO_TRAINING_SURFACE") or "").strip().lower()
    if surface == TrainingSurface.AGENT.value:
        return True
    return buzz_agent_present()


def resolve_training_surface(*, explicit: str | None = None) -> TrainingSurface:
    """Resolve the active training surface.

    Precedence: explicit argument → ``SEISO_TRAINING_SURFACE`` → agent env →
    frontend (safe default for Forge API / UI).
    """
    raw = (explicit or os.environ.get("SEISO_TRAINING_SURFACE") or "").strip().lower()
    if raw == TrainingSurface.AGENT.value:
        return TrainingSurface.AGENT
    if raw == TrainingSurface.FRONTEND.value:
        return TrainingSurface.FRONTEND
    if agent_context_present():
        return TrainingSurface.AGENT
    return TrainingSurface.FRONTEND


def require_buzz_agent(*, feature: str = "Mesh") -> None:
    """Refuse features that must only run under a Buzz agent identity."""
    if not buzz_agent_present():
        raise RuntimeError(
            f"{feature} is Buzz-agent-only. "
            "Configure BUZZ_PRIVATE_KEY (or BUZZ_AUTH_TAG in managed Desktop agents) "
            "and opt in with SEISO_ALLOW_MESH=1. "
            "Forge UI / frontend training cannot start mesh or multi-node plans."
        )
