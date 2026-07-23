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
    {
        "training",
        "export",
        "compress",
        "distill_rl",
        "rl_quant",
        "download",
        "experiment",
        "inference",
        "slime",
        "nemo_rl",
    }
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


def _gpu_resource_token(*, task: str, job_id: str | None = None, user_id: str | None = None) -> str:
    if job_id:
        return str(job_id)
    return f"{task}:{user_id or 'global'}:{uuid.uuid4().hex}"


def _register_gpu_task_if_available(
    *,
    task: str,
    job_id: str | None = None,
    user_id: str | None = None,
    gpu_resource_lock_held: bool = False,
) -> str:
    """Atomically reject tracked conflicts and reserve this process task."""
    with _GPU_TASK_LOCK:
        blocking = {
            str(active.get("task") or "gpu")
            for active in _ACTIVE_GPU_TASKS.values()
            if not job_id or active.get("job_id") != str(job_id)
        }
        if blocking:
            raise RuntimeError(
                "Another GPU task is still running "
                f"({', '.join(sorted(blocking))}). "
                "Finish or cancel it before starting a new one."
            )
        token = _gpu_resource_token(task=task, job_id=job_id, user_id=user_id)
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
        if (
            any(
                job.status == JobStatus.RUNNING and job.id != exclude_job_id
                for job in orch._jobs.values()
            )
            and orch.kind not in active
        ):
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

    # Optional managed multi-GPU vLLM holds external VRAM — stop on Free memory.
    managed_stopped = False
    try:
        from forge.config import get_settings
        from forge.services.managed_vllm import stop_managed_if_running

        settings = get_settings()
        managed = stop_managed_if_running(data_dir=settings.data_dir, reason=reason)
        managed_stopped = bool(managed.get("stopped"))
        if managed_stopped:
            note = "Stopped managed multi-GPU vLLM to free GPU memory"
            release_notes.append(note)
            if log:
                log(note)
            else:
                logger.info(note)
    except Exception:
        logger.debug("Managed vLLM stop during memory release skipped", exc_info=True)

    release_cached_memory(sync=had_active or managed_stopped)
    _refresh_hardware_profile()
    return {
        "unloaded_inference": unloaded,
        "previous_model": status_before.get("active_model"),
        "previous_path": status_before.get("path"),
        "release_notes": release_notes,
        "managed_vllm_stopped": managed_stopped,
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
        # Avoid a full device synchronize on the happy path; OOM recovery syncs explicitly.
        release_cached_memory(sync=False)
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
    try:
        resource_token = _register_gpu_task_if_available(
            task=task,
            job_id=job_id,
            user_id=user_id,
            gpu_resource_lock_held=True,
        )
    except Exception:
        from seiso.memory.gpu_resource_lock import release_gpu_resource_lock

        release_gpu_resource_lock()
        raise
    try:
        result = release_inference_memory(reason=task, log=log)
    except Exception:
        _unregister_gpu_task(resource_token=resource_token, job_id=job_id)
        from seiso.memory.gpu_resource_lock import release_gpu_resource_lock

        release_gpu_resource_lock()
        raise
    release_notes = [str(note) for note in result.get("release_notes") or []]
    sidecar_unload_uncertain = any(
        "Could not confirm llama-swap external model unload" in note or "Ollama unload" in note
        for note in release_notes
    )
    if sidecar_unload_uncertain:
        try:
            from seiso.inference.backends import _native_linux_requires_isolated_gguf

            native_linux_isolated = _native_linux_requires_isolated_gguf()
        except Exception:
            try:
                import platform

                native_linux_isolated = platform.system() == "Linux"
            except Exception:
                native_linux_isolated = True
        if native_linux_isolated:
            _unregister_gpu_task(resource_token=resource_token, job_id=job_id)
            from seiso.memory.gpu_resource_lock import release_gpu_resource_lock

            release_gpu_resource_lock()
            raise RuntimeError(
                "Could not confirm sidecar inference model unload; refusing to start "
                f"{task} until Ollama/llama-swap releases GPU memory."
            )
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
    try:
        from seiso.memory.gpu_resource_lock import (
            gpu_resource_lock_held_by_other_process,
        )

        locked_by_other = gpu_resource_lock_held_by_other_process()
    except Exception:
        locked_by_other = False
    if locked_by_other:
        raise RuntimeError(
            "Cannot load chat models while another Seiso GPU task is running. "
            "Wait for the task to finish before starting inference."
        )
    # Managed multi-GPU vLLM owns local GPUs; local in-process loads would OOM.
    try:
        from seiso.inference.managed_vllm import get_status

        status = get_status()
        if status.get("running") and status.get("managed"):
            raise RuntimeError(
                "Cannot load local chat models while managed multi-GPU vLLM is running. "
                "Use the Compat API model provider:<id> / Chat provider selector, "
                "or Free memory to stop managed vLLM."
            )
    except RuntimeError:
        raise
    except Exception:
        import logging

        logging.getLogger(__name__).debug(
            "Managed vLLM status check skipped during GPU gate", exc_info=True
        )
