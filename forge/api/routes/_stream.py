"""Shared SSE helpers for job log streams."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from forge.orchestrators.base import Orchestrator

logger = logging.getLogger(__name__)


def job_failure_message(
    orchestrator: Orchestrator, job_id: str, exc: BaseException | None = None
) -> str:
    """Best-effort failure text for DB persistence."""
    job = orchestrator.get_job(job_id)
    if job and job.error:
        return str(job.error)
    if exc is not None:
        return str(exc)
    return "Job failed"


def spawn_background(coro: Awaitable[Any]) -> asyncio.Task[Any]:
    """Run a coroutine in the background; log failures instead of re-raising."""

    async def _wrapper() -> Any:
        try:
            return await coro
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Background job failed")
            return None

    return asyncio.create_task(_wrapper())


async def job_log_event_gen(
    orchestrator: Orchestrator,
    job_id: str,
    *,
    before_result: Callable[[dict[str, Any]], list[dict[str, str]]] | None = None,
) -> AsyncIterator[dict[str, str]]:
    async for line in orchestrator.stream_logs(job_id):
        yield {"event": "log", "data": line}
    job = orchestrator.get_job(job_id)
    if job and job.error:
        yield {"event": "error", "data": job.error}
    if job and job.result:
        if before_result:
            for event in before_result(job.result):
                yield event
        yield {"event": "result", "data": json.dumps(job.result, default=str)}
