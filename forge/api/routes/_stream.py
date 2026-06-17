"""Shared SSE helpers for job log streams."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from typing import Any

from forge.orchestrators.base import Orchestrator


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
