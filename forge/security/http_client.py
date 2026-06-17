"""HTTP client helpers with DNS pinning to close SSRF rebinding windows."""

from __future__ import annotations

import socket
from contextlib import asynccontextmanager
from typing import Any

import httpx

from forge.security.url_policy import PinnedEndpoint


class _PinnedGetaddrinfo:
    """Force socket resolution for one hostname to a pre-validated IP."""

    def __init__(self, host: str, ip: str) -> None:
        self._host = host.lower().rstrip(".")
        self._ip = ip
        self._real = socket.getaddrinfo

    def __enter__(self) -> None:
        host = self._host
        ip = self._ip
        real = self._real

        def patched(name: str, *args: Any, **kwargs: Any) -> Any:
            if (name or "").lower().rstrip(".") == host:
                return real(ip, *args, **kwargs)
            return real(name, *args, **kwargs)

        socket.getaddrinfo = patched  # type: ignore[method-assign]

    def __exit__(self, *_: object) -> None:
        socket.getaddrinfo = self._real  # type: ignore[method-assign]


@asynccontextmanager
async def pinned_async_client(endpoint: PinnedEndpoint, *, timeout: float = 120.0):
    """Yield an httpx client that connects only to endpoint.pinned_ip (when set)."""
    if endpoint.pinned_ip:
        resolver = _PinnedGetaddrinfo(endpoint.host, endpoint.pinned_ip)
        resolver.__enter__()
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                yield client
        finally:
            resolver.__exit__()
    else:
        async with httpx.AsyncClient(timeout=timeout) as client:
            yield client


async def pinned_post(
    endpoint: PinnedEndpoint,
    path: str,
    *,
    headers: dict[str, str] | None = None,
    json: dict | None = None,
    timeout: float = 120.0,
) -> httpx.Response:
    url = f"{endpoint.base_url.rstrip('/')}{path}"
    async with pinned_async_client(endpoint, timeout=timeout) as client:
        return await client.post(url, headers=headers or {}, json=json)
