"""Resolve client IP for rate limits and audit logs."""

from __future__ import annotations

import ipaddress

from fastapi import Request

from forge.config import get_settings


def _normalize_ip(addr: str) -> str:
    try:
        return str(ipaddress.ip_address(addr.strip()))
    except ValueError:
        return addr.strip()


def _peer_ip(request: Request) -> str:
    if request.client and request.client.host:
        return _normalize_ip(request.client.host)
    return "unknown"


def _is_trusted_proxy(peer: str, trusted: list[str]) -> bool:
    if not trusted:
        return False
    try:
        peer_addr = ipaddress.ip_address(peer)
    except ValueError:
        return peer in trusted
    for entry in trusted:
        try:
            if "/" in entry:
                if peer_addr in ipaddress.ip_network(entry, strict=False):
                    return True
            elif peer_addr == ipaddress.ip_address(entry):
                return True
        except ValueError:
            if peer == entry:
                return True
    return False


def client_ip(request: Request) -> str:
    settings = get_settings()
    peer = _peer_ip(request)
    if settings.trust_proxy and _is_trusted_proxy(peer, settings.trusted_proxy_ip_list):
        trusted = settings.trusted_proxy_ip_list
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded:
            # Walk right-to-left: the nearest entries are appended by trusted
            # proxies, while leftmost entries are client-controlled spoof bait.
            for hop in reversed(forwarded.split(",")):
                candidate = _normalize_ip(hop)
                if not _is_trusted_proxy(candidate, trusted):
                    return candidate
            return peer
        real_ip = request.headers.get("x-real-ip", "").strip()
        if real_ip:
            return _normalize_ip(real_ip)
    return peer
