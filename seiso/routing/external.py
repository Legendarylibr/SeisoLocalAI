"""Localhost-only external router URL rules (shared with Forge)."""

from __future__ import annotations

import ipaddress
from urllib.parse import urlparse

LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def _host_is_loopback(host: str) -> bool:
    name = (host or "").strip().lower().rstrip(".")
    if name in LOOPBACK_HOSTS:
        return True
    if name.startswith("[") and name.endswith("]"):
        name = name[1:-1]
    try:
        return ipaddress.ip_address(name).is_loopback
    except ValueError:
        return False


def is_loopback_host(host: str | None) -> bool:
    """True for localhost, ::1, and 127.0.0.0/8."""
    if not host:
        return False
    return _host_is_loopback(host)


def is_loopback_url(url: str | None) -> bool:
    """True when *url* points at this machine (loopback).

    Bare ``localhost`` / ``127.0.0.1`` (no scheme) count as loopback so a
    mis-set ``SEISO_PAY_URL`` cannot bill self-hosted Forge.
    """
    raw = (url or "").strip()
    if not raw:
        return False
    parsed = urlparse(raw)
    if parsed.scheme and parsed.hostname:
        return is_loopback_host(parsed.hostname)
    if parsed.scheme and not parsed.hostname:
        return False
    # No scheme: treat the whole token or host:port as a hostname.
    host = raw.split("/", 1)[0]
    host = host.split(":", 1)[0]
    return is_loopback_host(host)


def validate_router_url(url: str) -> str:
    """Require an http(s) loopback router URL (SSRF-safe)."""
    raw = (url or "").strip()
    parsed = urlparse(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("model_router_url must be a valid http(s) URL")
    host = (parsed.hostname or "").lower()
    if not is_loopback_host(host):
        raise ValueError("model_router_url must point to localhost for local-first routing")
    return raw.rstrip("/")
