"""CSRF protection via double-submit cookie pattern."""

from __future__ import annotations

import secrets
from typing import Final

from fastapi import Request, Response

CSRF_COOKIE: Final = "seiso_csrf"
CSRF_HEADER: Final = "x-csrf-token"

# Pre-auth and health endpoints skip CSRF validation.
CSRF_EXEMPT_PATHS: Final = frozenset({
    "/api/auth/login",
    "/api/auth/register",
    "/api/auth/status",
    "/health",
    "/api/health",
})


def generate_csrf_token() -> str:
    return secrets.token_urlsafe(32)


def set_csrf_cookie(response: Response, token: str, *, secure: bool) -> None:
    response.set_cookie(
        CSRF_COOKIE,
        token,
        httponly=False,  # readable by SPA for double-submit header
        samesite="strict",
        secure=secure,
        max_age=86400 * 7,
    )


def clear_csrf_cookie(response: Response) -> None:
    response.delete_cookie(CSRF_COOKIE)


def validate_csrf(request: Request) -> bool:
    """Return True when the request passes CSRF checks."""
    if request.method not in ("POST", "PUT", "DELETE", "PATCH"):
        return True
    path = request.url.path
    if not path.startswith("/api") or path in CSRF_EXEMPT_PATHS:
        return True
    # Bearer-authenticated API clients are not vulnerable to browser CSRF.
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return True
    cookie_token = request.cookies.get(CSRF_COOKIE)
    header_token = request.headers.get(CSRF_HEADER)
    return bool(cookie_token and header_token and secrets.compare_digest(cookie_token, header_token))
