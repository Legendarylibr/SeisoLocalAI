"""Helpers for cookie-primary sessions (JWT body is opt-in for API clients)."""

from __future__ import annotations

from fastapi import Request

# Browser clients use HttpOnly cookies only. Non-browser clients that need a
# Bearer JWT (tests, CLI scripts) opt in with this header.
RETURN_TOKEN_HEADER = "X-Seiso-Return-Token"  # nosec B105 — header name, not a secret


def maybe_access_token(request: Request, token: str) -> str | None:
    """Return JWT for the response body only when the client opted in."""
    if request.headers.get(RETURN_TOKEN_HEADER, "").strip() == "1":
        return token
    return None
