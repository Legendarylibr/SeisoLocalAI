"""Cooperative cancellation for in-process training jobs."""

from __future__ import annotations

import threading
from collections.abc import Callable

_lock = threading.Lock()
_events: dict[str, threading.Event] = {}
# Cancel requests that arrive before register() stick until register consumes them.
_pending_requests: set[str] = set()


def register(job_id: str) -> None:
    with _lock:
        event = threading.Event()
        if job_id in _pending_requests:
            _pending_requests.discard(job_id)
            event.set()
        _events[job_id] = event


def clear(job_id: str) -> None:
    with _lock:
        _events.pop(job_id, None)
        _pending_requests.discard(job_id)


def request(job_id: str) -> None:
    with _lock:
        event = _events.get(job_id)
        if event is not None:
            event.set()
        else:
            # Sticky: cancel before the worker registers must not be lost.
            _pending_requests.add(job_id)


def is_requested(job_id: str) -> bool:
    with _lock:
        event = _events.get(job_id)
        if event is not None and event.is_set():
            return True
        return job_id in _pending_requests


def should_stop(job_id: str | None) -> Callable[[], bool]:
    if not job_id:
        return lambda: False
    return lambda: is_requested(job_id)
