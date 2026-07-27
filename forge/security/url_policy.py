"""Validate outbound provider URLs — block SSRF to private/metadata hosts."""

from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
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
# Local chat servers (loopback HTTP allowed). Canonical + legacy alias.
_LOCAL_CHAT_TYPES = frozenset({"local_chat", "vllm"})
# Documented defaults; any loopback port is also accepted for managed vLLM.
_LOCAL_DEFAULT_PORTS = {
    "local_chat": {8000, 8001},
    "vllm": {8000, 8001},
}
_ALLOW_ANY_LOOPBACK_PORT = True
# Remote multi-GPU chat servers (HTTPS only, no loopback). Canonical + legacy alias.
_REMOTE_CHAT_TYPES = frozenset({"remote_chat", "vllm_cloud"})
# CGNAT / shared-address space (RFC 6598) — not covered by ipaddress.is_private.
_BLOCKED_NETWORKS = (
    ipaddress.ip_network("100.64.0.0/10"),
)


def _literal_ip(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        return ipaddress.ip_address(host)
    except ValueError:
        return None


def _is_local_host(host: str) -> bool:
    if host in _LOCAL_HTTP_HOSTS:
        return True
    ip = _literal_ip(host)
    if ip is not None:
        return ip.is_loopback
    try:
        addrs = _resolve_host(host)
    except SecurityError:
        return False
    return bool(addrs) and all(
        (parsed := _literal_ip(addr)) is not None and parsed.is_loopback
        for addr in addrs
    )


def _is_blocked_ip(addr: str) -> bool:
    try:
        ip = ipaddress.ip_address(addr)
    except ValueError:
        return False
    if (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
        or str(ip) in _BLOCKED_HOSTS
    ):
        return True
    return any(ip in network for network in _BLOCKED_NETWORKS)


def _resolve_host(host: str) -> list[str]:
    if host in _BLOCKED_HOSTS:
        return [host]
    try:
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
        addrs = list({info[4][0] for info in infos})
        if not addrs:
            raise SecurityError("base_url host did not resolve to any address")
        return addrs
    except socket.gaierror as exc:
        raise SecurityError(f"base_url host could not be resolved: {host}") from exc


def validate_provider_base_url(url: str, *, provider_type: str = "local_chat") -> str:
    """Return normalized base URL or raise SecurityError."""
    raw = (url or "").strip()
    ptype = provider_type.lower()
    if not raw:
        if ptype in _LOCAL_CHAT_TYPES:
            return "http://127.0.0.1:8000"
        if ptype in _REMOTE_CHAT_TYPES:
            raise SecurityError(
                "remote_chat base_url is required (HTTPS remote chat server)"
            )
        raise SecurityError(f"Unsupported provider_type: {provider_type}")

    parsed = urlparse(raw)
    if not parsed.scheme or not parsed.netloc:
        raise SecurityError("Invalid base_url: missing scheme or host")
    if parsed.username or parsed.password:
        raise SecurityError("base_url must not include embedded credentials")

    host = (parsed.hostname or "").lower().rstrip(".")
    if not host:
        raise SecurityError("Invalid base_url: missing host")

    if host in _BLOCKED_HOSTS:
        raise SecurityError("base_url host is not allowed")

    scheme = parsed.scheme.lower()
    local_ok = ptype in _LOCAL_CHAT_TYPES and _is_local_host(host)
    remote_chat = ptype in _REMOTE_CHAT_TYPES

    if remote_chat and _is_local_host(host):
        raise SecurityError(
            "remote_chat base_url must be a remote HTTPS host "
            "(use type local_chat for loopback multi-GPU servers)"
        )

    literal = _literal_ip(host)
    if literal is not None and not local_ok and _is_blocked_ip(str(literal)):
        raise SecurityError("base_url host is not allowed")

    if scheme == "http" and not local_ok:
        raise SecurityError(
            "base_url must use HTTPS (http allowed only for local chat servers)"
        )
    if scheme not in ("http", "https"):
        raise SecurityError("base_url scheme must be http or https")
    if remote_chat and scheme != "https":
        raise SecurityError("remote_chat base_url must use HTTPS")

    if local_ok:
        port = parsed.port or 8000
        if port < 1 or port > 65535:
            raise SecurityError("Local chat server base_url port out of range")
        allowed = _LOCAL_DEFAULT_PORTS.get(ptype) or _LOCAL_DEFAULT_PORTS["local_chat"]
        if port not in allowed and not _ALLOW_ANY_LOOPBACK_PORT:
            raise SecurityError(
                f"Local chat server base_url must use port {sorted(allowed)}"
            )
    else:
        for addr in _resolve_host(host):
            if _is_blocked_ip(addr):
                raise SecurityError("base_url resolves to a blocked network range")

    port = parsed.port
    literal = _literal_ip(host)
    if literal is not None and literal.version == 6:
        bracketed = f"[{host}]"
        netloc = bracketed if port is None else f"{bracketed}:{port}"
    else:
        netloc = host if port is None else f"{host}:{port}"
    path = parsed.path.rstrip("/")
    return f"{scheme}://{netloc}{path}"


@dataclass(frozen=True)
class PinnedEndpoint:
    """Validated provider endpoint with optional DNS-pinned connect address."""

    base_url: str
    host: str
    port: int
    scheme: str
    pinned_ip: str | None


def resolve_pinned_endpoint(
    raw_url: str, *, provider_type: str = "local_chat"
) -> PinnedEndpoint:
    """Validate URL, resolve DNS, and return an endpoint pinned to the resolved IP."""
    base = validate_provider_base_url(raw_url, provider_type=provider_type).rstrip("/")
    parsed = urlparse(base)
    host = (parsed.hostname or "").lower().rstrip(".")
    scheme = parsed.scheme.lower()
    port = parsed.port or (443 if scheme == "https" else 80)
    ptype = provider_type.lower()
    # Only trust literal loopback / known local names without DNS pin.
    # DNS names that currently resolve to loopback must still be pinned so a
    # later rebinding cannot reach link-local/metadata after registration.
    literal = _literal_ip(host)
    local_literal = host in _LOCAL_HTTP_HOSTS or (
        literal is not None and literal.is_loopback
    )
    # Literal loopback / localhost need no DNS pin.
    if ptype in _LOCAL_CHAT_TYPES and local_literal:
        return PinnedEndpoint(
            base_url=base, host=host, port=port, scheme=scheme, pinned_ip=None
        )

    addrs = _resolve_host(host)
    all_loopback = bool(addrs) and all(
        (parsed_ip := _literal_ip(addr)) is not None and parsed_ip.is_loopback
        for addr in addrs
    )
    # DNS names that resolve only to loopback must still be pinned so a later
    # rebinding cannot reach link-local/metadata after registration.
    if ptype in _LOCAL_CHAT_TYPES and all_loopback:
        return PinnedEndpoint(
            base_url=base, host=host, port=port, scheme=scheme, pinned_ip=addrs[0]
        )

    for addr in addrs:
        if _is_blocked_ip(addr):
            raise SecurityError("base_url resolves to a blocked network range")

    return PinnedEndpoint(
        base_url=base, host=host, port=port, scheme=scheme, pinned_ip=addrs[0]
    )


def validate_nostr_relay_url(
    url: str,
    *,
    allowlist: list[str] | None = None,
    allow_loopback: bool = False,
) -> str:
    """Validate a Nostr relay URL (wss / loopback ws) against SSRF policy."""
    from seiso.research.nostr.policy import validate_relay_url

    return validate_relay_url(
        url, allowlist=allowlist, allow_loopback=allow_loopback
    )
