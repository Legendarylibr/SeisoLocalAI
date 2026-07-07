"""Shared executor pattern for bundled pipeline orchestrators."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

from forge.orchestrators.base import Orchestrator
from forge.services.user_paths import assert_user_path
from seiso.security import SecurityError


async def run_bundled_job(
    orchestrator: Orchestrator,
    job_id: str,
    payload: dict[str, Any],
    *,
    user_id_error: str,
    path_keys: tuple[str, ...],
    start_message: str,
    runner: Callable[..., dict[str, Any]],
    result_log: Callable[[dict[str, Any]], str],
) -> dict[str, Any]:
    from forge.services.memory_release import prepare_for_gpu_task, release_after_task

    user_id = payload.get("user_id")
    if not user_id:
        raise PermissionError(user_id_error)

    for key in path_keys:
        if path := payload.get(key):
            try:
                assert_user_path(orchestrator.sandbox_root, user_id, path)
            except SecurityError as exc:
                raise PermissionError(str(exc)) from exc

    prepare_for_gpu_task(
        task=orchestrator.kind,
        job_id=job_id,
        log=lambda msg: orchestrator._emit_log(job_id, msg),
    )
    orchestrator._emit_log(job_id, start_message)
    loop = asyncio.get_running_loop()

    def on_log(msg: str) -> None:
        loop.call_soon_threadsafe(orchestrator._emit_log, job_id, msg)

    try:
        result = await loop.run_in_executor(
            None,
            lambda: runner(
                job_id=job_id,
                user_id=user_id,
                data_dir=Path(orchestrator.sandbox_root),
                payload=payload,
                on_log=on_log,
            ),
        )
    finally:
        release_after_task(
            reason=f"{orchestrator.kind} complete",
            log=lambda msg: orchestrator._emit_log(job_id, msg),
            job_id=job_id,
        )

    orchestrator._emit_log(job_id, result_log(result))
    return result


def bundled_orchestrator(
    *,
    class_name: str,
    kind: str,
    user_id_error: str,
    path_keys: tuple[str, ...],
    start_message: str,
    runner: Callable[..., dict[str, Any]],
    result_log: Callable[[dict[str, Any]], str],
) -> type[Orchestrator]:
    """Build a thin Orchestrator subclass for a bundled pipeline runner."""

    class _BundledOrchestrator(Orchestrator):
        resource_key = "gpu"

        async def execute(self, job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
            return await run_bundled_job(
                self,
                job_id,
                payload,
                user_id_error=user_id_error,
                path_keys=path_keys,
                start_message=start_message,
                runner=runner,
                result_log=result_log,
            )

    _BundledOrchestrator.kind = kind
    _BundledOrchestrator.__name__ = class_name
    _BundledOrchestrator.__qualname__ = class_name
    return _BundledOrchestrator
