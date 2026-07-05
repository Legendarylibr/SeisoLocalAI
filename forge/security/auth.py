"""JWT auth, password hashing, rate limiting."""

from __future__ import annotations

import secrets
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from typing import Annotated, Deque, cast

import bcrypt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from forge.config import ForgeSettings, get_settings
from forge.security.token_revocation import is_jti_revoked, revoke_jti

bearer_scheme = HTTPBearer(auto_error=False)

ALGORITHM = "HS256"


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())


def create_access_token(
    subject: str,
    settings: ForgeSettings,
    *,
    hours: int | None = None,
) -> str:
    expire = datetime.now(timezone.utc) + timedelta(
        hours=hours or settings.session_hours
    )
    payload = {
        "sub": subject,
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "jti": secrets.token_hex(16),
    }
    return cast(str, jwt.encode(payload, settings.secret_key, algorithm=ALGORITHM))


def revoke_access_token(token: str, settings: ForgeSettings) -> None:
    """Invalidate a JWT so it cannot be reused after logout."""
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[ALGORITHM])
        jti = payload.get("jti")
        exp = payload.get("exp")
        if jti and exp is not None:
            revoke_jti(str(jti), float(exp))
    except JWTError:
        pass


class InvalidTokenError(Exception):
    """Raised when a token cannot be decoded or has been revoked."""


def decode_token(token: str, settings: ForgeSettings) -> str:
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[ALGORITHM])
        jti = payload.get("jti")
        if jti and is_jti_revoked(str(jti)):
            raise InvalidTokenError("Token revoked")
        sub = payload.get("sub")
        if not sub:
            raise InvalidTokenError("Invalid token")
        return str(sub)
    except JWTError as exc:
        raise InvalidTokenError("Invalid or expired token") from exc


async def get_current_user_id(
    request: Request,
    creds: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
    settings: Annotated[ForgeSettings, Depends(get_settings)],
) -> str:
    try:
        if creds and creds.credentials:
            return decode_token(creds.credentials, settings)
        cookie = request.cookies.get("seiso_token")
        if cookie:
            return decode_token(cookie, settings)
    except InvalidTokenError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, str(exc)) from exc
    raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Authentication required")


class RateLimiter:
    """Simple in-memory sliding window rate limiter per IP."""

    def __init__(self, max_per_minute: int = 120) -> None:
        self.max_per_minute = max_per_minute
        self._hits: dict[str, Deque[float]] = defaultdict(deque)

    def check(self, client_ip: str) -> None:
        now = time.monotonic()
        window = self._hits[client_ip]
        cutoff = now - 60.0
        while window and window[0] <= cutoff:
            window.popleft()
        if len(window) >= self.max_per_minute:
            raise HTTPException(
                status.HTTP_429_TOO_MANY_REQUESTS, "Rate limit exceeded"
            )
        window.append(now)


class LoginRateLimiter(RateLimiter):
    """Stricter limiter for authentication endpoints."""

    def __init__(self, max_per_minute: int = 10) -> None:
        super().__init__(max_per_minute=max_per_minute)
