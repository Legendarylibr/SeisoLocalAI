"""Scoped authentication for Compat API /v1 endpoints."""

from __future__ import annotations

import secrets
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials

from forge.api.deps import get_db
from forge.config import ForgeSettings, get_settings
from forge.db.store import Database
from forge.security.auth import InvalidTokenError, bearer_scheme, decode_token

_INFERENCE_KEY_PREFIX = "seiso_sk_"


async def get_compat_user_id(
    request: Request,
    creds: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
    settings: Annotated[ForgeSettings, Depends(get_settings)],
    db: Annotated[Database, Depends(get_db)],
) -> str:
    """Accept session JWT or inference-scoped API key (never full admin via API key alone)."""
    try:
        if creds and creds.credentials:
            token = creds.credentials.strip()
            if settings.inference_api_key and secrets.compare_digest(
                token, settings.inference_api_key
            ):
                user = await db.get_sole_user()
                if not user:
                    raise HTTPException(
                        status.HTTP_401_UNAUTHORIZED, "No local user configured"
                    )
                return str(user["id"])
            if token.startswith(_INFERENCE_KEY_PREFIX):
                raise HTTPException(
                    status.HTTP_401_UNAUTHORIZED,
                    "Invalid inference API key",
                )
            return decode_token(token, settings)

        cookie = request.cookies.get("seiso_token")
        if cookie:
            return decode_token(cookie, settings)
    except InvalidTokenError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, str(exc)) from exc
    raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Authentication required")
