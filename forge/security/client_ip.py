"""Resolve client IP for rate limits and audit logs."""

from __future__ import annotations

from fastapi import Request

from forge.config import get_settings


def client_ip(request: Request) -> str:
    settings = get_settings()
    if settings.trust_proxy:
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded:
            return forwarded.split(",")[0].strip()
        real_ip = request.headers.get("x-real-ip", "").strip()
        if real_ip:
            return real_ip
    return request.client.host if request.client else "unknown"
