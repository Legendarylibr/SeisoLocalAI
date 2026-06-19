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
_LOCAL_DEFAULT_PORTS = {"ollama": {11434}, "vllm": {8000, 8001}}


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
        (parsed := _literal_ip(addr)) is not None and parsed.is_loopback for addr in addrs
    )


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


def validate_provider_base_url(url: str, *, provider_type: str = "vllm") -> str:
    """Return normalized base URL or raise SecurityError."""
    raw = (url or "").strip()
    if not raw:
        ptype = provider_type.lower()
        if ptype == "ollama":
            return "http://127.0.0.1:11434"
        if ptype == "vllm":
            return "http://127.0.0.1:8000"
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
    ptype = provider_type.lower()
    local_ok = ptype in ("ollama", "vllm") and _is_local_host(host)

    literal = _literal_ip(host)
    if literal is not None and not local_ok and _is_blocked_ip(str(literal)):
        raise SecurityError("base_url host is not allowed")

    if scheme == "http" and not local_ok:
        raise SecurityError("base_url must use HTTPS (http allowed only for local ollama/vllm)")
    if scheme not in ("http", "https"):
        raise SecurityError("base_url scheme must be http or https")

    if local_ok:
        port = parsed.port or (11434 if ptype == "ollama" else 8000)
        allowed = _LOCAL_DEFAULT_PORTS.get(ptype, set())
        if port not in allowed:
            raise SecurityError(f"Local {ptype} base_url must use port {sorted(allowed)}")
    else:
        for addr in _resolve_host(host):
            if _is_blocked_ip(addr):
                raise SecurityError("base_url resolves to a blocked network range")

    port = parsed.port
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


def resolve_pinned_endpoint(raw_url: str, *, provider_type: str = "vllm") -> PinnedEndpoint:
    """Validate URL, resolve DNS, and return an endpoint pinned to the resolved IP."""
    base = validate_provider_base_url(raw_url, provider_type=provider_type).rstrip("/")
    parsed = urlparse(base)
    host = (parsed.hostname or "").lower().rstrip(".")
    scheme = parsed.scheme.lower()
    port = parsed.port or (443 if scheme == "https" else 80)
    ptype = provider_type.lower()
    local_ok = ptype in ("ollama", "vllm") and _is_local_host(host)

    if local_ok:
        return PinnedEndpoint(base_url=base, host=host, port=port, scheme=scheme, pinned_ip=None)

    addrs = _resolve_host(host)
    for addr in addrs:
        if _is_blocked_ip(addr):
            raise SecurityError("base_url resolves to a blocked network range")

    return PinnedEndpoint(base_url=base, host=host, port=port, scheme=scheme, pinned_ip=addrs[0])
