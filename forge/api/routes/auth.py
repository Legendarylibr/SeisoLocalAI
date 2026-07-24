"""Authentication routes."""

from __future__ import annotations

import os
from typing import Annotated, cast

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, Field

from forge.api.deps import clear_dependency_caches, close_dependency_caches, get_db
from forge.config import ForgeSettings, StorageMode, get_settings
from forge.db.store import Database
from forge.security.audit import audit_event
from forge.security.auth import (
    LoginRateLimiter,
    create_access_token,
    get_current_user_id,
    hash_password,
    revoke_access_token,
    verify_password,
)
from forge.security.client_ip import client_ip
from forge.security.csrf import (
    CSRF_COOKIE,
    clear_csrf_cookie,
    generate_csrf_token,
    set_csrf_cookie,
)
from seiso.security import generate_secret_key

_login_limiter = LoginRateLimiter()

DEFAULT_DISPLAY_NAME = "Admin"

router = APIRouter(prefix="/auth", tags=["auth"])


class RegisterRequest(BaseModel):
    # bcrypt truncates / rejects above 72 bytes — keep schema aligned.
    password: str = Field(min_length=8, max_length=72)
    storage_mode: str | None = Field(default=None, pattern="^(persistent|ephemeral)$")


class LoginRequest(BaseModel):
    password: str = Field(min_length=1, max_length=72)


class ResetSessionRequest(BaseModel):
    confirmation: str = Field(min_length=1, max_length=32)


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict


class OnboardingStatus(BaseModel):
    needs_onboarding: bool
    storage_mode: str
    storage_mode_configured: bool


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
    return OnboardingStatus(
        needs_onboarding=count == 0,
        storage_mode=settings.storage_mode,
        storage_mode_configured=settings.storage_mode_configured,
    )


@router.post(
    "/register", response_model=AuthResponse, status_code=status.HTTP_201_CREATED
)
async def register(
    body: RegisterRequest,
    response: Response,
    settings: Annotated[ForgeSettings, Depends(get_settings)],
) -> AuthResponse:
    if not settings.storage_mode_configured:
        if not body.storage_mode:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST, "Choose persistent or ephemeral storage"
            )
        settings.persist_storage_mode(cast(StorageMode, body.storage_mode))
        clear_dependency_caches()
    db = get_db()
    settings = get_settings()
    try:
        user = await db.create_first_user(
            hash_password(body.password), DEFAULT_DISPLAY_NAME
        )
    except ValueError as exc:
        raise HTTPException(status.HTTP_403_FORBIDDEN, str(exc)) from exc
    token = create_access_token(user["id"], settings)
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
    audit_event("auth_register", user_id=user["id"])
    return AuthResponse(
        access_token=token,
        user={
            "id": user["id"],
            "email": user["email"],
            "display_name": user["display_name"],
        },
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
    user = await db.get_sole_user()
    if not user or not verify_password(body.password, user["password_hash"]):
        audit_event("auth_login_failed")
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid credentials")

    token = create_access_token(user["id"], settings)
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
    audit_event("auth_login", user_id=user["id"])
    return AuthResponse(
        access_token=token,
        user={
            "id": user["id"],
            "email": user["email"],
            "display_name": user["display_name"],
        },
    )


@router.post("/reset-session")
async def reset_session(
    body: ResetSessionRequest,
    response: Response,
    db: Annotated[Database, Depends(get_db)],
    settings: Annotated[ForgeSettings, Depends(get_settings)],
) -> dict:
    """Reset a forgotten-password local instance back to onboarding."""
    if settings.allow_remote:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            "Password reset is only available on local-only Forge instances",
        )
    if body.confirmation.strip().upper() != "RESET":
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "Type RESET to confirm starting a new local session",
        )

    counts = await db.reset_local_session()
    sessions_rotated = False
    if "SEISO_SECRET_KEY" not in os.environ:
        key_file = settings.data_dir / ".secret_key"
        key_file.parent.mkdir(parents=True, exist_ok=True)
        key_file.write_text(generate_secret_key(), encoding="utf-8")
        key_file.chmod(0o600)
        sessions_rotated = True

    response.delete_cookie("seiso_token")
    clear_csrf_cookie(response)
    audit_event("auth_reset_session", rows_deleted=sum(counts.values()))
    await close_dependency_caches()
    clear_dependency_caches()
    return {
        "status": "reset",
        "needs_onboarding": True,
        "sessions_rotated": sessions_rotated,
        "rows_deleted": sum(counts.values()),
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
    response.delete_cookie("seiso_token")
    clear_csrf_cookie(response)
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
    return {
        "id": user["id"],
        "email": user["email"],
        "display_name": user["display_name"],
        "created_at": user["created_at"],
    }
