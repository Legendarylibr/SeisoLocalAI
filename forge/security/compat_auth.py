"""Scoped authentication for Compat API /v1 endpoints."""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from typing import Annotated, Literal

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials

from forge.api.deps import get_db
from forge.config import ForgeSettings, get_settings
from forge.db.store import Database
from forge.security.auth import InvalidTokenError, bearer_scheme, decode_token

_INFERENCE_KEY_PREFIX = "seiso_sk_"

CompatAuthMethod = Literal["inference_key", "session"]


@dataclass(frozen=True, slots=True)
class CompatIdentity:
    """Authenticated Compat caller with capability scope."""

    user_id: str
    auth_method: CompatAuthMethod

    @property
    def tools_allowed(self) -> bool:
        """Inference API key is chat-only; session JWT may use Compat tools when enabled."""
        return self.auth_method == "session"


async def get_compat_identity(
    request: Request,
    creds: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
    settings: Annotated[ForgeSettings, Depends(get_settings)],
    db: Annotated[Database, Depends(get_db)],
) -> CompatIdentity:
    """Accept session JWT/cookie or inference-scoped API key (never full admin via API key alone)."""
    try:
        if creds and creds.credentials:
            token = creds.credentials.strip()
            expected_key = settings.inference_api_key or ""
            # compare_digest raises on length mismatch — only compare equal lengths.
            if (
                expected_key
                and len(token) == len(expected_key)
                and secrets.compare_digest(token, expected_key)
            ):
                user = await db.get_sole_user()
                if not user:
                    raise HTTPException(status.HTTP_401_UNAUTHORIZED, "No local user configured")
                owner = settings.get_inference_api_key_owner()
                user_pubkey = str(user.get("nostr_pubkey") or "").strip().lower()
                if owner:
                    if not user_pubkey or owner != user_pubkey:
                        raise HTTPException(
                            status.HTTP_401_UNAUTHORIZED,
                            "Inference API key is not bound to the current owner npub",
                        )
                elif user_pubkey:
                    # Legacy installs: bind existing key to the sole owner npub.
                    settings.bind_inference_api_key_owner(user_pubkey)
                identity = CompatIdentity(user_id=str(user["id"]), auth_method="inference_key")
                request.state.compat_auth_method = identity.auth_method
                return identity
            if token.startswith(_INFERENCE_KEY_PREFIX):
                raise HTTPException(
                    status.HTTP_401_UNAUTHORIZED,
                    "Invalid inference API key",
                )
            user_id = decode_token(token, settings)
            auth_method: CompatAuthMethod = "session"
        else:
            cookie = request.cookies.get("seiso_token")
            if not cookie:
                raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Authentication required")
            user_id = decode_token(cookie, settings)
            auth_method = "session"
    except InvalidTokenError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, str(exc)) from exc

    # Mirror /api auth: reject JWTs for deleted/wiped users (ghost tokens).
    user = await db.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Authentication required")
    identity = CompatIdentity(user_id=user_id, auth_method=auth_method)
    request.state.compat_auth_method = identity.auth_method
    return identity


async def get_compat_user_id(
    identity: Annotated[CompatIdentity, Depends(get_compat_identity)],
) -> str:
    """Backward-compatible user-id dependency for Compat routes."""
    return identity.user_id
