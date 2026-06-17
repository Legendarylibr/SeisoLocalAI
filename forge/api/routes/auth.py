"""Authentication routes."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, EmailStr, Field

from forge.api.deps import get_db
from forge.config import ForgeSettings, get_settings
from forge.db.store import Database
from forge.security.audit import audit_event
from forge.security.auth import (
    LoginRateLimiter,
    create_access_token,
    get_current_user_id,
    hash_password,
    verify_password,
)
from forge.security.csrf import clear_csrf_cookie, generate_csrf_token, set_csrf_cookie

_login_limiter = LoginRateLimiter()

router = APIRouter(prefix="/auth", tags=["auth"])


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    display_name: str | None = Field(default=None, max_length=64)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


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

    existing = await db.get_user_by_email(body.email)
    if existing:
        raise HTTPException(status.HTTP_409_CONFLICT, "Email already registered")

    user = await db.create_user(body.email, hash_password(body.password), body.display_name)
    token = create_access_token(user["id"], settings)
    csrf = generate_csrf_token()
    response.set_cookie(
        "seiso_token",
        token,
        httponly=True,
        samesite="strict",
        secure=settings.allow_remote,
        max_age=settings.session_hours * 3600,
    )
    set_csrf_cookie(response, csrf, secure=settings.allow_remote)
    audit_event("auth_register", user_id=user["id"], email=body.email)
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
    client = request.client.host if request.client else "unknown"
    _login_limiter.check(client)
    user = await db.get_user_by_email(body.email)
    if not user or not verify_password(body.password, user["password_hash"]):
        audit_event("auth_login_failed", email=body.email)
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid credentials")

    token = create_access_token(user["id"], settings)
    csrf = generate_csrf_token()
    response.set_cookie(
        "seiso_token",
        token,
        httponly=True,
        samesite="strict",
        secure=settings.allow_remote,
        max_age=settings.session_hours * 3600,
    )
    set_csrf_cookie(response, csrf, secure=settings.allow_remote)
    audit_event("auth_login", user_id=user["id"], email=body.email)
    return AuthResponse(
        access_token=token,
        user={"id": user["id"], "email": user["email"], "display_name": user["display_name"]},
    )


@router.post("/logout")
async def logout(response: Response) -> dict[str, str]:
    response.delete_cookie("seiso_token")
    clear_csrf_cookie(response)
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
