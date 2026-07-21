"""Shared guards for inference generation routes.

Used by both the native chat routes (``inference.py``) and the
Compat API routes (``compat.py``) so the GPU-availability check and the
per-user generation gate stay in a single place.
"""

from __future__ import annotations

from fastapi import HTTPException

from forge.orchestrators.inference import InferenceOrchestrator


def _assert_inference_gpu_available() -> None:
    from forge.services.memory_release import assert_gpu_available_for_inference

    try:
        assert_gpu_available_for_inference()
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc


def _begin_generation_or_raise(
    orchestrator: InferenceOrchestrator,
    user_id: str | None,
) -> int:
    try:
        return orchestrator.begin_generation_for_user(user_id)
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
