"""Shared executor pattern for vendored pipeline orchestrators."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

from forge.orchestrators.base import Orchestrator
from forge.services.user_paths import assert_user_path
from seiso.security import SecurityError


async def run_vendor_job(
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
    user_id = payload.get("user_id")
    if not user_id:
        raise PermissionError(user_id_error)

    for key in path_keys:
        if path := payload.get(key):
            try:
                assert_user_path(orchestrator.sandbox_root, user_id, path)
            except SecurityError as exc:
                raise PermissionError(str(exc)) from exc

    orchestrator._emit_log(job_id, start_message)
    loop = asyncio.get_running_loop()

    def on_log(msg: str) -> None:
        loop.call_soon_threadsafe(orchestrator._emit_log, job_id, msg)

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
    orchestrator._emit_log(job_id, result_log(result))
    return result


def vendor_orchestrator(
    *,
    class_name: str,
    kind: str,
    user_id_error: str,
    path_keys: tuple[str, ...],
    start_message: str,
    runner: Callable[..., dict[str, Any]],
    result_log: Callable[[dict[str, Any]], str],
) -> type[Orchestrator]:
    """Build a thin Orchestrator subclass for a vendored pipeline runner."""

    class _VendorOrchestrator(Orchestrator):
        async def execute(self, job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
            return await run_vendor_job(
                self,
                job_id,
                payload,
                user_id_error=user_id_error,
                path_keys=path_keys,
                start_message=start_message,
                runner=runner,
                result_log=result_log,
            )

    _VendorOrchestrator.kind = kind
    _VendorOrchestrator.__name__ = class_name
    _VendorOrchestrator.__qualname__ = class_name
    return _VendorOrchestrator
