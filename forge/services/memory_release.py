"""Unified VRAM/RAM release before and after Forge GPU-heavy work."""

from __future__ import annotations

import logging
import threading
import uuid
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

LogFn = Callable[[str], None] | None

_GPU_TASK_KINDS = frozenset(
    {"training", "export", "compress", "distill_rl", "rl_quant", "download"}
)
_GPU_TASK_LOCK = threading.RLock()
_ACTIVE_GPU_TASKS: dict[str, dict[str, str | None]] = {}
_GPU_RESOURCE_LOCK_HELD = "gpu_resource_lock_held"


def _refresh_hardware_profile() -> None:
    try:
        from seiso.hardware.profile import hardware_profile

        hardware_profile(force_refresh=True)
    except ImportError:
        pass


def _gpu_resource_token(
    *, task: str, job_id: str | None = None, user_id: str | None = None
) -> str:
    if job_id:
        return str(job_id)
    return f"{task}:{user_id or 'global'}:{uuid.uuid4().hex}"


def _register_gpu_task(
    *,
    task: str,
    job_id: str | None = None,
    user_id: str | None = None,
    gpu_resource_lock_held: bool = False,
) -> str:
    token = _gpu_resource_token(task=task, job_id=job_id, user_id=user_id)
    with _GPU_TASK_LOCK:
        _ACTIVE_GPU_TASKS[token] = {
            "task": task,
            "job_id": str(job_id) if job_id else None,
            "user_id": str(user_id) if user_id else None,
            _GPU_RESOURCE_LOCK_HELD: "1" if gpu_resource_lock_held else None,
        }
    return token


def _unregister_gpu_task(
    *, resource_token: str | None = None, job_id: str | None = None
) -> dict[str, str | None] | None:
    with _GPU_TASK_LOCK:
        if resource_token:
            return _ACTIVE_GPU_TASKS.pop(resource_token, None)
        if job_id:
            return _ACTIVE_GPU_TASKS.pop(str(job_id), None)
    return None


def _active_tracked_gpu_task_kinds(*, exclude_job_id: str | None = None) -> list[str]:
    with _GPU_TASK_LOCK:
        tasks = list(_ACTIVE_GPU_TASKS.values())
    active: list[str] = []
    for task in tasks:
        if exclude_job_id and task.get("job_id") == str(exclude_job_id):
            continue
        kind = str(task.get("task") or "gpu")
        if kind not in active:
            active.append(kind)
    return active


def running_gpu_task_kinds(*, exclude_job_id: str | None = None) -> list[str]:
    """Return active GPU-heavy orchestrator and service task kinds."""
    from forge.orchestrators.base import JobStatus

    try:
        from forge.api import deps
    except ImportError:
        return _active_tracked_gpu_task_kinds(exclude_job_id=exclude_job_id)

    getters = [
        deps.get_training_orchestrator,
        deps.get_export_orchestrator,
        deps.get_compress_orchestrator,
        deps.get_distill_rl_orchestrator,
        deps.get_rl_quant_orchestrator,
    ]
    active: list[str] = _active_tracked_gpu_task_kinds(exclude_job_id=exclude_job_id)
    for getter in getters:
        orch = getter()
        if any(
            job.status == JobStatus.RUNNING and job.id != exclude_job_id
            for job in orch._jobs.values()
        ):
            if orch.kind not in active:
                active.append(orch.kind)
    return active


def release_inference_memory(*, reason: str, log: LogFn = None) -> dict[str, Any]:
    """Unload the active chat/inference model and clear GPU/RAM caches."""
    from seiso.inference.model_pool import get_model_pool
    from seiso.memory.protection import release_cached_memory

    pool = get_model_pool()
    if hasattr(pool, "drain_release_notes"):
        pool.drain_release_notes()
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
    if hasattr(pool, "drain_release_notes"):
        release_notes = pool.drain_release_notes()
    else:
        status_after = pool.status()
        release_notes = list(status_after.get("release_notes") or [])
    for note in release_notes:
        if log:
            log(note)
        else:
            logger.info(note)
    release_cached_memory(sync=had_active)
    _refresh_hardware_profile()
    return {
        "unloaded_inference": unloaded,
        "previous_model": status_before.get("active_model"),
        "previous_path": status_before.get("path"),
        "release_notes": release_notes,
    }


def release_after_task(
    *,
    reason: str,
    log: LogFn = None,
    job_id: str | None = None,
    resource_token: str | None = None,
) -> None:
    """Post-task cleanup: restore kernel patches and empty GPU caches."""
    from seiso.kernels.lifecycle import restore_kernel_patches
    from seiso.memory.gpu_resource_lock import release_gpu_resource_lock
    from seiso.memory.protection import release_cached_memory

    task = None
    try:
        restore_kernel_patches()
        release_cached_memory(sync=True)
        _refresh_hardware_profile()
    finally:
        task = _unregister_gpu_task(resource_token=resource_token, job_id=job_id)
        if task and task.get(_GPU_RESOURCE_LOCK_HELD) == "1":
            release_gpu_resource_lock()
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
    task = str(task)
    if task not in _GPU_TASK_KINDS:
        logger.warning("Unregistered GPU task kind requested memory prep: %s", task)
    blocking = running_gpu_task_kinds(exclude_job_id=job_id)
    if blocking:
        msg = (
            f"Another GPU task is still running ({', '.join(blocking)}). "
            "Finish or cancel it before starting a new one."
        )
        if log:
            log(msg)
        raise RuntimeError(msg)
    try:
        from forge.api import deps

        inference = deps.get_inference_orchestrator()
        inference.assert_generation_available_for_user(None)
        inference.assert_backend_idle()
    except ImportError:
        pass
    except PermissionError as exc:
        raise RuntimeError(str(exc)) from exc
    from seiso.memory.gpu_resource_lock import acquire_gpu_resource_lock

    acquire_gpu_resource_lock()
    resource_token = _register_gpu_task(
        task=task,
        job_id=job_id,
        user_id=user_id,
        gpu_resource_lock_held=True,
    )
    try:
        result = release_inference_memory(reason=task, log=log)
    except Exception:
        _unregister_gpu_task(resource_token=resource_token, job_id=job_id)
        from seiso.memory.gpu_resource_lock import release_gpu_resource_lock

        release_gpu_resource_lock()
        raise
    result["resource_token"] = resource_token
    return result


def assert_gpu_available_for_inference() -> None:
    """Raise when a GPU-heavy Forge job would conflict with loading chat models."""
    blocking = running_gpu_task_kinds()
    if blocking:
        raise RuntimeError(
            f"Cannot load chat models while {', '.join(blocking)} is running. "
            "Wait for the job to finish or cancel it first."
        )
