"""HTTP client for the Seiso model router service."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import urlparse

import httpx

from forge.config import ForgeSettings

logger = logging.getLogger(__name__)

ROUTER_MODEL_ID = "__seiso_router__"


def _validate_router_url(url: str) -> str:
    parsed = urlparse(url.strip())
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("model_router_url must be a valid http(s) URL")
    host = (parsed.hostname or "").lower()
    if host not in {"127.0.0.1", "localhost", "::1"} and not host.endswith(".internal"):
        raise ValueError("model_router_url must point to localhost for local-first routing")
    return url.rstrip("/")


def router_headers(settings: ForgeSettings) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    key = (settings.model_router_api_key or "").strip()
    if key:
        headers["Authorization"] = f"Bearer {key}"
        headers["X-API-Key"] = key
    return headers


async def fetch_router_status(settings: ForgeSettings) -> dict[str, Any]:
    base = _validate_router_url(settings.model_router_url)
    timeout = httpx.Timeout(10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        health = await client.get(f"{base}/health")
        status: dict[str, Any] = {"health": health.json() if health.status_code == 200 else {}}
        try:
            detail = await client.get(f"{base}/router/status")
            if detail.status_code == 200:
                status["detail"] = detail.json()
        except Exception as exc:
            logger.debug("router status unavailable: %s", exc)
        return status


async def router_chat_completion(
    settings: ForgeSettings,
    messages: list[dict[str, Any]],
    *,
    model: str | None = None,
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> tuple[str, dict[str, Any]]:
    base = _validate_router_url(settings.model_router_url)
    payload: dict[str, Any] = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    if model:
        payload["model"] = model

    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
        resp = await client.post(
            f"{base}/v1/chat/completions",
            json=payload,
            headers=router_headers(settings),
        )
        resp.raise_for_status()
        data = resp.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
        meta = data.get("seiso_router") or {}
        return content, meta


async def router_stream_chat(
    settings: ForgeSettings,
    messages: list[dict[str, Any]],
    *,
    model: str | None = None,
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> AsyncIterator[str]:
    base = _validate_router_url(settings.model_router_url)
    payload: dict[str, Any] = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }
    if model:
        payload["model"] = model

    timeout = httpx.Timeout(None)
    async with (
        httpx.AsyncClient(timeout=timeout) as client,
        client.stream(
            "POST",
            f"{base}/v1/chat/completions",
            json=payload,
            headers=router_headers(settings),
        ) as resp,
    ):
        if resp.status_code >= 400:
            body = await resp.aread()
            raise RuntimeError(body.decode("utf-8", errors="replace") or "Router request failed")
        async for line in resp.aiter_lines():
            if not line.startswith("data:"):
                continue
            raw = line[5:].strip()
            if not raw or raw == "[DONE]":
                continue
            try:
                chunk = json.loads(raw)
            except json.JSONDecodeError:
                continue
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            token = delta.get("content")
            if token:
                yield token
