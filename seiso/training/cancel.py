"""Cooperative cancellation for in-process training jobs."""

from __future__ import annotations

import threading
from collections.abc import Callable

_lock = threading.Lock()
_events: dict[str, threading.Event] = {}


def register(job_id: str) -> None:
    with _lock:
        _events[job_id] = threading.Event()


def clear(job_id: str) -> None:
    with _lock:
        _events.pop(job_id, None)


def request(job_id: str) -> None:
    with _lock:
        event = _events.get(job_id)
        if event is not None:
            event.set()


def is_requested(job_id: str) -> bool:
    with _lock:
        event = _events.get(job_id)
        return bool(event and event.is_set())


def should_stop(job_id: str | None) -> Callable[[], bool]:
    if not job_id:
        return lambda: False
    return lambda: is_requested(job_id)
