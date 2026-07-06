"""Unified VRAM/RAM release before and after Forge GPU-heavy work."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

LogFn = Callable[[str], None] | None

_GPU_TASK_KINDS = frozenset(
    {"training", "export", "compress", "distill_rl", "rl_quant", "download"}
)


def _refresh_hardware_profile() -> None:
    try:
        from seiso.hardware.profile import hardware_profile

        hardware_profile(force_refresh=True)
    except ImportError:
        pass


def running_gpu_task_kinds(*, exclude_job_id: str | None = None) -> list[str]:
    """Return in-process GPU-heavy orchestrator kinds with RUNNING jobs."""
    from forge.orchestrators.base import JobStatus

    try:
        from forge.api import deps
    except ImportError:
        return []

    getters = [
        deps.get_training_orchestrator,
        deps.get_export_orchestrator,
        deps.get_compress_orchestrator,
        deps.get_distill_rl_orchestrator,
        deps.get_rl_quant_orchestrator,
    ]
    active: list[str] = []
    for getter in getters:
        orch = getter()
        if any(
            job.status == JobStatus.RUNNING and job.id != exclude_job_id
            for job in orch._jobs.values()
        ):
            active.append(orch.kind)
    return active


def release_inference_memory(*, reason: str, log: LogFn = None) -> dict[str, Any]:
    """Unload the active chat/inference model and clear GPU/RAM caches."""
    from seiso.inference.model_pool import get_model_pool
    from seiso.memory.protection import release_cached_memory

    pool = get_model_pool()
    status_before = pool.status()
    had_active = bool(pool.active_key)
    if had_active:
        msg = f"Unloading active inference model to free memory ({reason})"
        if log:
            log(msg)
        else:
            logger.info(msg)
        # Wait for in-flight inference so VRAM is actually freed before GPU tasks.
        pool.prepare_for_load()

    unloaded = pool.active_key is None and had_active
    release_cached_memory(sync=had_active)
    _refresh_hardware_profile()
    return {
        "unloaded_inference": unloaded,
        "previous_model": status_before.get("active_model"),
        "previous_path": status_before.get("path"),
    }


def release_after_task(*, reason: str, log: LogFn = None) -> None:
    """Post-task cleanup: restore kernel patches and empty GPU caches."""
    from seiso.kernels.lifecycle import restore_kernel_patches
    from seiso.memory.protection import release_cached_memory

    restore_kernel_patches()
    release_cached_memory(sync=True)
    _refresh_hardware_profile()
    if log:
        log(f"Released GPU/RAM caches ({reason})")


def prepare_for_gpu_task(
    *,
    task: str,
    log: LogFn = None,
    job_id: str | None = None,
    user_id: str | None = None,
) -> dict[str, Any]:
    """Eject inference weights and refresh headroom before a heavy local task."""
    blocking = running_gpu_task_kinds(exclude_job_id=job_id)
    if blocking:
        msg = (
            f"Another GPU task is still running ({', '.join(blocking)}). "
            "Finish or cancel it before starting a new one."
        )
        if log:
            log(msg)
        raise RuntimeError(msg)
    if user_id is not None:
        try:
            from forge.api import deps

            deps.get_inference_orchestrator().assert_generation_available_for_user(
                user_id
            )
        except PermissionError as exc:
            raise RuntimeError(str(exc)) from exc
    return release_inference_memory(reason=task, log=log)


def assert_gpu_available_for_inference() -> None:
    """Raise when a GPU-heavy Forge job would conflict with loading chat models."""
    blocking = running_gpu_task_kinds()
    if blocking:
        raise RuntimeError(
            f"Cannot load chat models while {', '.join(blocking)} is running. "
            "Wait for the job to finish or cancel it first."
        )
