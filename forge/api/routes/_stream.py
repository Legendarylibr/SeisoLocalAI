"""Shared SSE helpers for job log streams."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Callable
from typing import Any

from forge.db.store import Database
from forge.orchestrators.base import Orchestrator
from forge.services.job_runtime import (
    job_failure_message as job_failure_message,
)
from forge.services.job_runtime import (
    spawn_background as spawn_background,
)

logger = logging.getLogger(__name__)

__all__ = [
    "durable_job_events",
    "job_failure_message",
    "job_log_event_gen",
    "spawn_background",
]


async def job_log_event_gen(
    orchestrator: Orchestrator,
    job_id: str,
    *,
    db: Database | None = None,
    user_id: str | None = None,
    before_result: Callable[[dict[str, Any]], list[dict[str, str]]] | None = None,
) -> AsyncIterator[dict[str, str]]:
    live_job = orchestrator.get_job(job_id)
    replay_durable = db is not None and user_id is not None and live_job is None
    if replay_durable and db is not None and user_id is not None:
        async for event in durable_job_events(db, job_id, user_id):
            yield event
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


async def durable_job_events(
    db: Database,
    job_id: str,
    user_id: str,
    *,
    event_types: tuple[str, ...] = (
        "log",
        "metric",
        "error",
        "result",
        "status",
        "policy",
        "memory_policy",
    ),
) -> AsyncIterator[dict[str, str]]:
    """Replay persisted job events in SSE shape."""
    rows = await db.list_job_events(job_id, user_id, event_types=event_types)
    for row in rows:
        event_type = str(row.get("event_type") or "message")
        payload = row.get("payload") or {}
        if event_type == "log":
            yield {"event": "log", "data": str(payload.get("line", ""))}
        elif event_type == "metric":
            yield {"event": "metric", "data": json.dumps(payload, default=str)}
        elif event_type == "status":
            yield {"event": "status", "data": json.dumps(payload, default=str)}
        elif event_type == "result":
            yield {"event": "result", "data": json.dumps(payload, default=str)}
        elif event_type in {"policy", "memory_policy"}:
            yield {"event": event_type, "data": json.dumps(payload, default=str)}
        elif event_type == "error":
            yield {"event": "error", "data": str(payload.get("error", ""))}
