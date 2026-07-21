"""HTTP client helpers with DNS pinning to close SSRF rebinding windows."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import httpx

from forge.security.url_policy import PinnedEndpoint


def _normalize_host(host: str) -> str:
    return (host or "").lower().rstrip(".")


def pin_request_to_ip(
    request: httpx.Request, *, host: str, pinned_ip: str
) -> httpx.Request:
    """Rewrite a request to connect via ``pinned_ip`` without global DNS patches.

    Preserves the original hostname in the ``Host`` header and TLS SNI so
    certificate verification still targets the validated hostname while the TCP
    connect address stays on the pre-resolved IP (SSRF DNS-rebinding defense).
    """
    expected = _normalize_host(host)
    req_host = _normalize_host(request.url.host or "")
    if not expected or req_host != expected:
        return request

    headers = request.headers.copy()
    headers["host"] = expected
    extensions = dict(request.extensions)
    extensions["sni_hostname"] = expected
    return httpx.Request(
        method=request.method,
        url=request.url.copy_with(host=pinned_ip),
        headers=headers,
        stream=request.stream,
        extensions=extensions,
    )


class _PinnedIPTransport(httpx.AsyncHTTPTransport):
    """httpx transport that connects via a pre-validated IP for one hostname."""

    def __init__(self, host: str, pinned_ip: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._host = _normalize_host(host)
        self._pinned_ip = pinned_ip

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        pinned = pin_request_to_ip(
            request, host=self._host, pinned_ip=self._pinned_ip
        )
        return await super().handle_async_request(pinned)


@asynccontextmanager
async def pinned_async_client(endpoint: PinnedEndpoint, *, timeout: float = 120.0):
    """Yield an httpx client that connects only to endpoint.pinned_ip (when set)."""
    if endpoint.pinned_ip:
<<<<<<< Updated upstream
        resolver = _PinnedGetaddrinfo(endpoint.host, endpoint.pinned_ip)
        resolver.__enter__()
        try:
            async with httpx.AsyncClient(
                timeout=timeout, follow_redirects=False
            ) as client:
                yield client
        finally:
            resolver.__exit__()
    else:
        async with httpx.AsyncClient(
            timeout=timeout, follow_redirects=False
=======
        transport = _PinnedIPTransport(endpoint.host, endpoint.pinned_ip)
        async with httpx.AsyncClient(
            transport=transport,
            timeout=timeout,
            follow_redirects=False,
        ) as client:
            yield client
    else:
        async with httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=False,
>>>>>>> Stashed changes
        ) as client:
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
