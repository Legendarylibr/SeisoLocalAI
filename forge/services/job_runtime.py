"""Shared orchestration lifecycle helpers for Forge jobs."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from forge.orchestrators.base import JobRecord, Orchestrator

logger = logging.getLogger(__name__)


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


def job_failure_message(
    orchestrator: Orchestrator, job_id: str, exc: BaseException | None = None
) -> str:
    """Best-effort failure text for durable job status persistence."""
    job = orchestrator.get_job(job_id)
    if job and job.error:
        return str(job.error)
    if exc is not None:
        return str(exc)
    return "Job failed"


async def run_orchestrated_job(
    *,
    orchestrator: Orchestrator,
    job_id: str,
    payload: dict[str, Any],
    on_finished: Callable[[JobRecord], Awaitable[None]],
    on_failed: Callable[[str], Awaitable[None]],
    on_started: Callable[[], Awaitable[None]] | None = None,
) -> None:
    """Start an orchestrator job and persist a deterministic terminal state."""
    try:
        if on_started is not None:
            await on_started()
        await orchestrator.start(job_id, payload)
        job = await orchestrator.wait_for(job_id)
    except asyncio.CancelledError:
        # Persist cancel even when the outer task is cancelled during release.
        job = orchestrator.get_job(job_id)
        if job is not None:
            job.cancel_requested = True
            from forge.orchestrators.base import JobStatus

            if job.status not in (
                JobStatus.CANCELLED,
                JobStatus.COMPLETED,
                JobStatus.FAILED,
            ):
                job.status = JobStatus.CANCELLED
            try:
                await on_finished(job)
            except Exception:
                logger.exception("Failed to persist cancelled state for job %s", job_id)
        raise
    except Exception as exc:
        await on_failed(job_failure_message(orchestrator, job_id, exc))
        return

    if job is None:
        await on_failed("Job disappeared before completion")
        return
    try:
        await on_finished(job)
    except Exception:
        # Never rewrite a terminal in-memory success/cancel as failed because
        # durable persistence threw — artifacts already exist.
        logger.exception(
            "Failed to persist terminal state for job %s (status=%s)",
            job_id,
            getattr(job.status, "value", job.status),
        )
