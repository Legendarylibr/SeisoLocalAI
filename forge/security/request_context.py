"""Per-request correlation context for audit trails."""

from __future__ import annotations

from contextvars import ContextVar

_REQUEST_ID: ContextVar[str | None] = ContextVar("seiso_request_id", default=None)

REQUEST_ID_HEADER = "X-Request-ID"


def get_request_id() -> str | None:
    return _REQUEST_ID.get()


def set_request_id(request_id: str | None):
    """Bind request_id for the current async/task context; returns a reset token."""
    return _REQUEST_ID.set(request_id)


def reset_request_id(token) -> None:
    _REQUEST_ID.reset(token)
