"""Outbound Nostr gates and relay URL validation (SSRF-safe)."""

from __future__ import annotations

import ipaddress
import os
import socket
from urllib.parse import urlparse

from seiso.security import SecurityError

_BLOCKED_HOSTS = frozenset(
    {
        "metadata.google.internal",
        "169.254.169.254",
        "metadata.azure.com",
    }
)
_LOCAL_HTTP_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def _env_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def nostr_allowed() -> bool:
    """Master kill-switch; default off."""
    return _env_enabled("SEISO_ALLOW_NOSTR")


def nostr_auto_attest_enabled() -> bool:
    return nostr_allowed() and _env_enabled("SEISO_NOSTR_ATTEST")


def relay_allowlist_from_env() -> list[str]:
    raw = os.environ.get("SEISO_NOSTR_RELAYS", "").strip()
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def _literal_ip(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        return ipaddress.ip_address(host)
    except ValueError:
        return None


def _is_blocked_ip(addr: str) -> bool:
    try:
        ip = ipaddress.ip_address(addr)
    except ValueError:
        return False
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
        or str(ip) in _BLOCKED_HOSTS
    )


def _is_local_host(host: str) -> bool:
    if host in _LOCAL_HTTP_HOSTS:
        return True
    ip = _literal_ip(host)
    return ip is not None and ip.is_loopback


def _resolve_host(host: str) -> list[str]:
    if host in _BLOCKED_HOSTS:
        return [host]
    try:
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
        addrs = list(dict.fromkeys(str(info[4][0]) for info in infos))
        if not addrs:
            raise SecurityError(f"relay host did not resolve: {host}")
        return addrs
    except socket.gaierror as exc:
        raise SecurityError(f"relay host could not be resolved: {host}") from exc


def validate_relay_url(
    url: str,
    *,
    allowlist: list[str] | None = None,
    allow_loopback: bool = False,
) -> str:
    """Return normalized relay URL or raise SecurityError.

    - ``wss://`` required for remote hosts
    - ``ws://`` only for loopback when ``allow_loopback`` is true
    - private / link-local / metadata ranges blocked
    - when ``allowlist`` is non-empty, hostname must match exactly
    """
    raw = (url or "").strip()
    if not raw:
        raise SecurityError("relay URL is required")
    parsed = urlparse(raw)
    if not parsed.scheme or not parsed.netloc:
        raise SecurityError("invalid relay URL: missing scheme or host")
    if parsed.username or parsed.password:
        raise SecurityError("relay URL must not include embedded credentials")
    host = (parsed.hostname or "").lower().rstrip(".")
    if not host:
        raise SecurityError("invalid relay URL: missing host")
    if host in _BLOCKED_HOSTS:
        raise SecurityError("relay host is not allowed")

    scheme = parsed.scheme.lower()
    local = _is_local_host(host)
    if scheme == "ws":
        if not (allow_loopback and local):
            raise SecurityError("ws:// is only allowed for loopback relays")
    elif scheme != "wss":
        raise SecurityError("relay URL scheme must be wss (or ws for loopback)")

    if allowlist:
        normalized_allow = {h.lower().rstrip(".") for h in allowlist if h.strip()}
        if host not in normalized_allow:
            raise SecurityError(f"relay host {host!r} is not in the allowlist")

    literal = _literal_ip(host)
    if literal is not None:
        if local and allow_loopback:
            pass
        elif _is_blocked_ip(str(literal)):
            raise SecurityError("relay host is not allowed")
    elif not local:
        for addr in _resolve_host(host):
            if _is_blocked_ip(addr):
                raise SecurityError("relay resolves to a blocked network range")
    elif not allow_loopback:
        raise SecurityError("loopback relays require allow_loopback=True")

    port = parsed.port
    if literal is not None and literal.version == 6:
        bracketed = f"[{host}]"
        netloc = bracketed if port is None else f"{bracketed}:{port}"
    else:
        netloc = host if port is None else f"{host}:{port}"
    path = parsed.path.rstrip("/")
    return f"{scheme}://{netloc}{path}"


def normalize_relay_list(
    relays: list[str],
    *,
    allowlist: list[str] | None = None,
    allow_loopback: bool = False,
) -> list[str]:
    if not relays:
        raise SecurityError("at least one relay URL is required")
    out: list[str] = []
    seen: set[str] = set()
    for raw in relays:
        normalized = validate_relay_url(
            raw, allowlist=allowlist, allow_loopback=allow_loopback
        )
        if normalized not in seen:
            seen.add(normalized)
            out.append(normalized)
    return out
