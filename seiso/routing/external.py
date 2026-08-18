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


def _hostport_without_scheme(raw: str) -> str:
    """Extract host from a scheme-less token.

    ``urlparse("127.0.0.1:8787")`` treats the IPv4 address as a scheme, so
    callers must not feed bare ``host:port`` values through ``urlparse``.
    """
    host = raw.split("/", 1)[0]
    if host.startswith("["):
        end = host.find("]")
        return host[1:end] if end != -1 else host[1:]
    # IPv4 or hostname with optional :port. Leave IPv6 (multiple colons) intact.
    if host.count(":") == 1:
        return host.split(":", 1)[0]
    return host


def is_loopback_url(url: str | None) -> bool:
    """True when *url* points at this machine (loopback).

    Bare ``localhost`` / ``127.0.0.1`` (no scheme) count as loopback so a
    mis-set ``SEISO_PAY_URL`` cannot bill self-hosted Forge.
    """
    raw = (url or "").strip()
    if not raw:
        return False
    if "://" not in raw:
        return is_loopback_host(_hostport_without_scheme(raw))
    parsed = urlparse(raw)
    if parsed.hostname:
        return is_loopback_host(parsed.hostname)
    return False


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
