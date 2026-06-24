"""Middleware for production router."""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from collections.abc import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger(__name__)


class APIKeyMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, api_keys: list[str]) -> None:
        super().__init__(app)
        self._keys = {k.strip() for k in api_keys if k.strip()}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not self._keys:
            return await call_next(request)
        if request.url.path in ("/health", "/metrics", "/ready"):
            return await call_next(request)

        token = ""
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            token = auth[7:].strip()
        if not token:
            token = request.headers.get("x-api-key", "").strip()

        if token not in self._keys:
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory token bucket per client IP."""

    def __init__(self, app, rpm: int, burst: int) -> None:
        super().__init__(app)
        self._rpm = rpm
        self._burst = burst
        self._buckets: dict[str, tuple[float, float]] = defaultdict(lambda: (0.0, float(burst)))

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if self._rpm <= 0:
            return await call_next(request)
        if request.url.path in ("/health", "/metrics", "/ready"):
            return await call_next(request)

        client = request.client.host if request.client else "unknown"
        now = time.monotonic()
        tokens, last = self._buckets[client]
        refill = (now - last) * (self._rpm / 60.0)
        tokens = min(float(self._burst), tokens + refill)
        if tokens < 1.0:
            return JSONResponse({"error": "rate_limit_exceeded"}, status_code=429)
        self._buckets[client] = (tokens - 1.0, now)
        return await call_next(request)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, json_logs: bool = False) -> None:
        super().__init__(app)
        self._json = json_logs

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        msg = (
            f"router {request.method} {request.url.path} "
            f"status={response.status_code} latency_ms={elapsed_ms:.1f}"
        )
        if self._json:
            logger.info(
                msg,
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "latency_ms": elapsed_ms,
                },
            )
        else:
            logger.info(msg)
        return response
