"""JWT auth, password hashing, rate limiting."""

from __future__ import annotations

import secrets
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Annotated

import bcrypt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from forge.config import ForgeSettings, get_settings

bearer_scheme = HTTPBearer(auto_error=False)

ALGORITHM = "HS256"
_revoked_jtis: set[str] = set()


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
    expire = datetime.now(timezone.utc) + timedelta(hours=hours or settings.session_hours)
    payload = {
        "sub": subject,
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "jti": secrets.token_hex(16),
    }
    return jwt.encode(payload, settings.secret_key, algorithm=ALGORITHM)


def revoke_access_token(token: str, settings: ForgeSettings) -> None:
    """Invalidate a JWT so it cannot be reused after logout."""
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[ALGORITHM])
        jti = payload.get("jti")
        if jti:
            _revoked_jtis.add(str(jti))
    except JWTError:
        pass


def decode_token(token: str, settings: ForgeSettings) -> str:
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[ALGORITHM])
        jti = payload.get("jti")
        if jti and str(jti) in _revoked_jtis:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Token revoked")
        sub = payload.get("sub")
        if not sub:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid token")
        return str(sub)
    except JWTError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired token") from exc


async def get_current_user_id(
    request: Request,
    creds: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
    settings: Annotated[ForgeSettings, Depends(get_settings)],
) -> str:
    if creds and creds.credentials:
        return decode_token(creds.credentials, settings)
    # Cookie fallback for browser sessions
    cookie = request.cookies.get("seiso_token")
    if cookie:
        return decode_token(cookie, settings)
    raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Authentication required")


class RateLimiter:
    """Simple in-memory sliding window rate limiter per IP."""

    def __init__(self, max_per_minute: int = 120) -> None:
        self.max_per_minute = max_per_minute
        self._hits: dict[str, list[float]] = defaultdict(list)

    def check(self, client_ip: str) -> None:
        now = time.monotonic()
        window = self._hits[client_ip]
        cutoff = now - 60.0
        self._hits[client_ip] = [t for t in window if t > cutoff]
        if len(self._hits[client_ip]) >= self.max_per_minute:
            raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "Rate limit exceeded")
        self._hits[client_ip].append(now)


class LoginRateLimiter(RateLimiter):
    """Stricter limiter for authentication endpoints."""

    def __init__(self, max_per_minute: int = 10) -> None:
        super().__init__(max_per_minute=max_per_minute)
