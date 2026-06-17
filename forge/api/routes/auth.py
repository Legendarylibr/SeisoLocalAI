"""Authentication routes."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, Field

from forge.api.deps import get_db
from forge.config import ForgeSettings, get_settings
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
from forge.security.csrf import clear_csrf_cookie, generate_csrf_token, set_csrf_cookie

_login_limiter = LoginRateLimiter()

DEFAULT_DISPLAY_NAME = "Admin"

router = APIRouter(prefix="/auth", tags=["auth"])


class RegisterRequest(BaseModel):
    password: str = Field(min_length=8, max_length=128)


class LoginRequest(BaseModel):
    password: str = Field(min_length=1, max_length=128)


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict


class OnboardingStatus(BaseModel):
    needs_onboarding: bool


@router.get("/status", response_model=OnboardingStatus)
async def onboarding_status(db: Annotated[Database, Depends(get_db)]) -> OnboardingStatus:
    count = await db.user_count()
    return OnboardingStatus(needs_onboarding=count == 0)


@router.post("/register", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
async def register(
    body: RegisterRequest,
    response: Response,
    db: Annotated[Database, Depends(get_db)],
    settings: Annotated[ForgeSettings, Depends(get_settings)],
) -> AuthResponse:
    count = await db.user_count()
    if count > 0:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Registration closed — user already exists")

    user = await db.create_user(hash_password(body.password), DEFAULT_DISPLAY_NAME)
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
        user={"id": user["id"], "email": user["email"], "display_name": user["display_name"]},
    )


@router.post("/login", response_model=AuthResponse)
async def login(
    body: LoginRequest,
    request: Request,
    response: Response,
    db: Annotated[Database, Depends(get_db)],
    settings: Annotated[ForgeSettings, Depends(get_settings)],
) -> AuthResponse:
    if settings.rate_limit_enabled:
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
        user={"id": user["id"], "email": user["email"], "display_name": user["display_name"]},
    )


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
    user_id: Annotated[str, Depends(get_current_user_id)],
    db: Annotated[Database, Depends(get_db)],
) -> dict:
    user = await db.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "User not found")
    return {
        "id": user["id"],
        "email": user["email"],
        "display_name": user["display_name"],
        "created_at": user["created_at"],
    }
