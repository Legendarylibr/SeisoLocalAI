"""Durable job event persistence for restart-safe streams."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from forge.db.store import Database

logger = logging.getLogger(__name__)


class DurableJobEventSink:
    """Fire-and-forget bridge from synchronous orchestrator emits to SQLite."""

    def __init__(self, db: Database) -> None:
        self._db = db

    def emit(
        self,
        *,
        job_id: str,
        user_id: str,
        kind: str,
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.debug("Dropping %s event for %s outside event loop", event_type, job_id)
            return
        task = loop.create_task(
            self._db.append_job_event(
                job_id=job_id,
                user_id=user_id,
                kind=kind,
                event_type=event_type,
                payload=payload,
            )
        )
        task.add_done_callback(_log_persist_failure)


def _log_persist_failure(task: asyncio.Task[object]) -> None:
    try:
        task.result()
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("Failed to persist job event")
