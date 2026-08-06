"""CSRF protection via double-submit cookie pattern."""

from __future__ import annotations

import secrets
from typing import Final

from fastapi import Request, Response

CSRF_COOKIE: Final = "seiso_csrf"
CSRF_HEADER: Final = "x-csrf-token"

# Pre-auth and health endpoints skip CSRF validation.
CSRF_EXEMPT_PATHS: Final = frozenset(
    {
        "/api/auth/login",
        "/api/auth/register",
        # reset-session requires CSRF (cookie issued by GET /api/auth/status).
        "/api/auth/status",
        "/health",
        "/api/health",
    }
)

# Mutating routes outside /api and /v1 that skip CSRF validation. Default-deny:
# every current mutating route lives under /api or /v1, so this set is empty —
# add a path here only with a documented reason.
CSRF_NON_API_EXEMPT_PATHS: Final[frozenset[str]] = frozenset()


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


def clear_csrf_cookie(response: Response, *, secure: bool = False) -> None:
    # Must match set_csrf_cookie attributes or browsers keep Secure cookies.
    response.delete_cookie(CSRF_COOKIE, path="/", samesite="strict", secure=secure)


def validate_csrf(request: Request) -> bool:
    """Return True when the request passes CSRF checks."""
    if request.method not in ("POST", "PUT", "DELETE", "PATCH"):
        return True
    path = request.url.path
    if path in CSRF_EXEMPT_PATHS:
        return True
    # Bearer-authenticated API clients are not vulnerable to browser CSRF.
    # Only skip for a valid session JWT or the configured inference API key
    # (S1-010) — junk Bearer text must not bypass double-submit cookie checks.
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer ") and auth[7:].strip():
        from forge.config import get_settings
        from forge.security.auth import InvalidTokenError, decode_token

        token = auth[7:].strip()
        settings = get_settings()
        try:
            decode_token(token, settings)
            return True
        except InvalidTokenError:
            pass
        expected_key = settings.inference_api_key or ""
        # compare_digest raises on length mismatch — only compare equal lengths.
        if (
            expected_key
            and len(token) == len(expected_key)
            and secrets.compare_digest(token, expected_key)
        ):
            return True
    if not (path.startswith("/api") or path.startswith("/v1")):
        return path in CSRF_NON_API_EXEMPT_PATHS
    cookie_token = request.cookies.get(CSRF_COOKIE)
    header_token = request.headers.get(CSRF_HEADER)
    return bool(
        cookie_token and header_token and secrets.compare_digest(cookie_token, header_token)
    )
