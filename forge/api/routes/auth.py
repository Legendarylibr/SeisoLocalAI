"""Authentication routes — Nostr nsec proves ownership of npub identity."""

from __future__ import annotations

import os
from typing import Annotated, cast

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, ConfigDict, Field, model_validator

from forge.api.deps import clear_dependency_caches, close_dependency_caches, get_db
from forge.config import ForgeSettings, StorageMode, get_settings
from forge.db.store import Database
from forge.security.audit import audit_event
from forge.security.auth import (
    LoginRateLimiter,
    create_access_token,
    get_current_user_id,
    revoke_access_token,
)
from forge.security.client_ip import client_ip
from forge.security.csrf import (
    CSRF_COOKIE,
    clear_csrf_cookie,
    generate_csrf_token,
    set_csrf_cookie,
)
from forge.security.session_token import maybe_access_token
from forge.services.nostr_auth import (
    NOSTR_PASSWORD_SENTINEL,
    npub_from_pubkey_hex,
    persist_user_signing_key,
    resolve_identity,
    user_public_view,
)
from seiso.security import generate_secret_key

_login_limiter = LoginRateLimiter()

DEFAULT_DISPLAY_NAME = "Admin"

router = APIRouter(prefix="/auth", tags=["auth"])


class RegisterRequest(BaseModel):
    """First-time setup: generate a key (default) or import an nsec."""

    model_config = ConfigDict(extra="forbid")

    nsec: str | None = Field(default=None, max_length=256)
    # None = default path (generate when nsec omitted; import when nsec set).
    generate: bool | None = None
    storage_mode: str | None = Field(default=None, pattern="^(persistent|ephemeral)$")

    @model_validator(mode="after")
    def _require_key_source(self) -> RegisterRequest:
        has_nsec = bool(self.nsec and self.nsec.strip())
        generate = self.generate
        if generate is None:
            generate = not has_nsec
        if generate and has_nsec:
            raise ValueError("Pass either generate=true or nsec, not both")
        if not generate and not has_nsec:
            raise ValueError("Provide nsec or set generate=true")
        self.generate = generate
        return self


class LoginRequest(BaseModel):
    """Sign in by proving possession of the account nsec."""

    model_config = ConfigDict(extra="forbid")

    nsec: str = Field(min_length=8, max_length=256)


class ResetSessionRequest(BaseModel):
    confirmation: str = Field(min_length=1, max_length=32)


class AuthResponse(BaseModel):
    # Cookie session is primary; body JWT only when X-Seiso-Return-Token: 1.
    access_token: str | None = None
    token_type: str = "bearer"
    user: dict
    # Returned only when Forge generated a fresh key during register.
    nsec: str | None = None


class OnboardingStatus(BaseModel):
    needs_onboarding: bool
    storage_mode: str
    storage_mode_configured: bool
    auth_method: str = "nostr"
    # Instance owner npub when an account exists (public identity).
    owner_npub: str | None = None


def _set_session_cookies(
    response: Response, token: str, settings: ForgeSettings
) -> None:
    csrf = generate_csrf_token()
    response.set_cookie(
        "seiso_token",
        token,
        httponly=True,
        samesite="strict",
        secure=settings.cookie_secure,
        max_age=settings.session_hours * 3600,
    )
    set_csrf_cookie(response, csrf, secure=settings.cookie_secure)


@router.get("/status", response_model=OnboardingStatus)
async def onboarding_status(
    request: Request,
    response: Response,
    db: Annotated[Database, Depends(get_db)],
) -> OnboardingStatus:
    count = await db.user_count()
    settings = get_settings()
    # Issue CSRF before reset-session / register so pre-auth forms can double-submit.
    if not request.cookies.get(CSRF_COOKIE):
        set_csrf_cookie(response, generate_csrf_token(), secure=settings.cookie_secure)
    owner_npub: str | None = None
    if count > 0:
        sole = await db.get_sole_user()
        pubkey = str((sole or {}).get("nostr_pubkey") or "").strip()
        if len(pubkey) == 64:
            owner_npub = npub_from_pubkey_hex(pubkey)
    return OnboardingStatus(
        needs_onboarding=count == 0,
        storage_mode=settings.storage_mode,
        storage_mode_configured=settings.storage_mode_configured,
        auth_method="nostr",
        owner_npub=owner_npub,
    )


@router.post(
    "/register", response_model=AuthResponse, status_code=status.HTTP_201_CREATED
)
async def register(
    body: RegisterRequest,
    request: Request,
    response: Response,
    settings: Annotated[ForgeSettings, Depends(get_settings)],
) -> AuthResponse:
    _login_limiter.check(client_ip(request))
    if not settings.storage_mode_configured:
        if not body.storage_mode:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST, "Choose persistent or ephemeral storage"
            )
        settings.persist_storage_mode(cast(StorageMode, body.storage_mode))
        clear_dependency_caches()
    db = get_db()
    settings = get_settings()
    generate = bool(body.generate)
    try:
        identity = resolve_identity(nsec=body.nsec, generate=generate)
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    try:
        user = await db.create_first_user(
            NOSTR_PASSWORD_SENTINEL,
            DEFAULT_DISPLAY_NAME,
            nostr_pubkey=identity.pubkey_hex,
        )
    except ValueError as exc:
        raise HTTPException(status.HTTP_403_FORBIDDEN, str(exc)) from exc

    persist_user_signing_key(
        data_dir=settings.data_dir,
        user_id=user["id"],
        pair=identity.pair,
        persist=not settings.db_ephemeral,
    )
    # Compat /v1 key is owned by this npub (rotate if unbound or prior owner).
    settings.sync_inference_api_key_owner(identity.pubkey_hex)
    token = create_access_token(user["id"], settings)
    _set_session_cookies(response, token, settings)
    audit_event(
        "auth_register",
        user_id=user["id"],
        nostr_pubkey=identity.pubkey_hex,
        generated=generate,
    )
    return AuthResponse(
        access_token=maybe_access_token(request, token),
        user=user_public_view(user),
        nsec=identity.nsec if generate else None,
    )


@router.post("/login", response_model=AuthResponse)
async def login(
    body: LoginRequest,
    request: Request,
    response: Response,
    db: Annotated[Database, Depends(get_db)],
    settings: Annotated[ForgeSettings, Depends(get_settings)],
) -> AuthResponse:
    _login_limiter.check(client_ip(request))
    try:
        identity = resolve_identity(nsec=body.nsec, generate=False)
    except ValueError:
        audit_event("auth_login_failed")
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid credentials") from None

    user = await db.get_sole_user()
    stored = str((user or {}).get("nostr_pubkey") or "").strip().lower()
    if not user or not stored or stored != identity.pubkey_hex:
        audit_event("auth_login_failed")
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid credentials")

    # Refresh encrypted signing key (same nsec) for provenance attest.
    persist_user_signing_key(
        data_dir=settings.data_dir,
        user_id=user["id"],
        pair=identity.pair,
        persist=not settings.db_ephemeral,
    )
    # Ensure Compat key stays bound to the logging-in owner npub.
    settings.sync_inference_api_key_owner(identity.pubkey_hex)
    token = create_access_token(user["id"], settings)
    _set_session_cookies(response, token, settings)
    audit_event(
        "auth_login", user_id=user["id"], nostr_pubkey=identity.pubkey_hex
    )
    return AuthResponse(
        access_token=maybe_access_token(request, token),
        user=user_public_view(user),
    )


@router.post("/reset-session")
async def reset_session(
    body: ResetSessionRequest,
    response: Response,
    db: Annotated[Database, Depends(get_db)],
    settings: Annotated[ForgeSettings, Depends(get_settings)],
) -> dict:
    """Reset a forgotten-key local instance back to onboarding."""
    if settings.allow_remote:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            "Session reset is only available on local-only Forge instances",
        )
    if body.confirmation.strip().upper() != "RESET":
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "Type RESET to confirm starting a new local session",
        )

    # Fail closed before wiping anything when the Compat key cannot rotate
    # (env-bound). Otherwise a prior /v1 holder keeps working for the next
    # npub after forgotten-key wipe.
    if "SEISO_INFERENCE_API_KEY" in os.environ:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "Session reset refused: SEISO_INFERENCE_API_KEY is env-bound and "
            "cannot rotate. Unset it (or use a disk-managed key) before wipe.",
        )

    counts = await db.reset_local_session()
    from forge.services.nostr_settings import wipe_nostr_identity_material

    nostr_wipe = wipe_nostr_identity_material(settings.data_dir)
    sessions_rotated = False
    if "SEISO_SECRET_KEY" not in os.environ:
        key_file = settings.data_dir / ".secret_key"
        key_file.parent.mkdir(parents=True, exist_ok=True)
        key_file.write_text(generate_secret_key(), encoding="utf-8")
        key_file.chmod(0o600)
        sessions_rotated = True
    # Drop owner binding + Compat key so a prior key cannot authenticate as
    # the next npub after forgotten-key wipe + re-onboard.
    settings.clear_inference_api_key_owner()
    inference_key_rotated = settings.rotate_inference_api_key()
    if not inference_key_rotated:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "Session reset refused: Compat /v1 key could not be rotated",
        )

    response.delete_cookie(
        "seiso_token",
        path="/",
        samesite="strict",
        secure=settings.cookie_secure,
    )
    clear_csrf_cookie(response, secure=settings.cookie_secure)
    audit_event(
        "auth_reset_session",
        rows_deleted=sum(counts.values()),
        nostr_keys_removed=nostr_wipe.get("removed_files"),
        nostr_encryption_key_rotated=nostr_wipe.get("encryption_key_rotated"),
        inference_key_rotated=inference_key_rotated,
        owner_cleared=True,
    )
    await close_dependency_caches()
    clear_dependency_caches()
    return {
        "status": "reset",
        "needs_onboarding": True,
        "sessions_rotated": sessions_rotated,
        "inference_key_rotated": inference_key_rotated,
        "owner_cleared": True,
        "owner_npub": None,
        "rows_deleted": sum(counts.values()),
        "nostr_identity_wiped": True,
    }


@router.post("/logout")
async def logout(
    request: Request,
    response: Response,
    user_id: Annotated[str, Depends(get_current_user_id)],
    db: Annotated[Database, Depends(get_db)],
    settings: Annotated[ForgeSettings, Depends(get_settings)],
) -> dict[str, str]:
    token = request.cookies.get("seiso_token")
    if not token:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.lower().startswith("bearer "):
            token = auth_header[7:].strip()
    if token:
        revoke_access_token(token, settings)
    await db.purge_user_chat(user_id)
    response.delete_cookie(
        "seiso_token",
        path="/",
        samesite="strict",
        secure=settings.cookie_secure,
    )
    clear_csrf_cookie(response, secure=settings.cookie_secure)
    audit_event("auth_logout", user_id=user_id)
    return {"status": "ok"}


@router.get("/me")
async def me(
    request: Request,
    response: Response,
    user_id: Annotated[str, Depends(get_current_user_id)],
    db: Annotated[Database, Depends(get_db)],
    settings: Annotated[ForgeSettings, Depends(get_settings)],
) -> dict:
    user = await db.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "User not found")
    # Issue CSRF cookie for existing sessions (e.g. after upgrade) so POST/SSE calls work.
    if not request.cookies.get(CSRF_COOKIE):
        set_csrf_cookie(response, generate_csrf_token(), secure=settings.cookie_secure)
    return user_public_view(user)
