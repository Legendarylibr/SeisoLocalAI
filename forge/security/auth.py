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
from forge.security.token_revocation import is_jti_revoked, revoke_jti

bearer_scheme = HTTPBearer(auto_error=False)

ALGORITHM = "HS256"


_BCRYPT_MAX_BYTES = 72


def _password_bytes(password: str) -> bytes:
    raw = password.encode()
    if len(raw) > _BCRYPT_MAX_BYTES:
        raise ValueError(
            f"Password must be at most {_BCRYPT_MAX_BYTES} bytes (bcrypt limit)"
        )
    return raw


def hash_password(password: str) -> str:
    return bcrypt.hashpw(_password_bytes(password), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(_password_bytes(plain), hashed.encode())


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
    return jwt.encode(payload, settings.secret_key, algorithm=ALGORITHM)


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
    """Decode the session JWT and require a live DB user (blocks ghost tokens)."""
    from forge.api.deps import get_db

    try:
        if creds and creds.credentials:
            user_id = decode_token(creds.credentials, settings)
        else:
            cookie = request.cookies.get("seiso_token")
            if not cookie:
                raise HTTPException(
                    status.HTTP_401_UNAUTHORIZED, "Authentication required"
                )
            user_id = decode_token(cookie, settings)
    except InvalidTokenError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, str(exc)) from exc

    user = await get_db().get_user_by_id(user_id)
    if not user:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Authentication required")
    return user_id


class RateLimiter:
    """Simple in-memory sliding window rate limiter per IP."""

    def __init__(self, max_per_minute: int = 120) -> None:
        self.max_per_minute = max_per_minute
        self._hits: dict[str, list[float]] = defaultdict(list)

    def check(self, client_ip: str) -> None:
        now = time.monotonic()
        cutoff = now - 60.0
        pruned = [t for t in self._hits.get(client_ip, []) if t > cutoff]
        if len(pruned) >= self.max_per_minute:
            self._hits[client_ip] = pruned
            raise HTTPException(
                status.HTTP_429_TOO_MANY_REQUESTS, "Rate limit exceeded"
            )
        pruned.append(now)
        self._hits[client_ip] = pruned


class LoginRateLimiter(RateLimiter):
    """Stricter limiter for authentication endpoints."""

    def __init__(self, max_per_minute: int = 10) -> None:
        super().__init__(max_per_minute=max_per_minute)
