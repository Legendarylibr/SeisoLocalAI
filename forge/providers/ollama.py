"""Local Ollama HTTP client."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from forge.security.http_client import pinned_async_client, pinned_post
from forge.security.url_policy import resolve_pinned_endpoint


def _endpoint(base_url: str = ""):
    return resolve_pinned_endpoint(base_url, provider_type="ollama")


def _chat_path(endpoint) -> str:
    base = endpoint.base_url.rstrip("/")
    if base.endswith("/v1"):
        return "/chat/completions"
    return "/v1/chat/completions"


async def ensure_model_available(model: str, base_url: str = "") -> str:
    """Return a local Ollama tag matching ``model`` or raise if missing."""
    names = [m["name"] for m in await list_models(base_url) if m.get("name")]
    if model in names:
        return model
    base = model.split(":")[0]
    for name in names:
        if name.split(":")[0] == base:
            return name
    raise ValueError(
        f"Ollama model {model!r} is not available locally. "
        "Run `ollama pull` or create the model from your GGUF, then retry."
    )


async def warm_model(model: str, base_url: str = "") -> None:
    """Load ``model`` into the Ollama runtime with a tiny generation."""
    tag = await ensure_model_available(model, base_url)
    endpoint = _endpoint(base_url)
    base = endpoint.base_url.rstrip("/")
    url = f"{base}/api/chat" if not base.endswith("/v1") else f"{base.rsplit('/v1', 1)[0]}/api/chat"
    payload = {
        "model": tag,
        "messages": [{"role": "user", "content": "ping"}],
        "stream": False,
        "options": {"num_predict": 1},
    }
    async with pinned_async_client(endpoint, timeout=300.0) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()


async def unload_model(model: str, base_url: str = "") -> None:
    """Ask Ollama to evict a resident model from memory."""
    endpoint = _endpoint(base_url)
    base = endpoint.base_url.rstrip("/")
    url = (
        f"{base}/api/generate"
        if not base.endswith("/v1")
        else f"{base.rsplit('/v1', 1)[0]}/api/generate"
    )
    payload = {
        "model": model,
        "prompt": "",
        "stream": False,
        "keep_alive": 0,
    }
    async with pinned_async_client(endpoint, timeout=60.0) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()


async def list_models(base_url: str = "") -> list[dict[str, Any]]:
    """Return models from Ollama ``GET /api/tags``."""
    endpoint = _endpoint(base_url)
    tags_path = "/api/tags"
    if endpoint.base_url.rstrip("/").endswith("/v1"):
        tags_path = "/../api/tags"
    url = f"{endpoint.base_url.rstrip('/')}{tags_path}".replace("/v1/../", "/")
    async with pinned_async_client(endpoint, timeout=15.0) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        data = resp.json()
    models = data.get("models", [])
    return [
        {
            "name": m.get("name", ""),
            "size_bytes": m.get("size", 0),
            "modified_at": m.get("modified_at"),
            "family": (m.get("details") or {}).get("family"),
        }
        for m in models
        if m.get("name")
    ]


async def chat_completion(
    messages: list[dict],
    *,
    model: str,
    max_tokens: int = 512,
    base_url: str = "",
) -> str:
    endpoint = _endpoint(base_url)
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": False,
    }
    resp = await pinned_post(endpoint, _chat_path(endpoint), json=payload, timeout=300.0)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


async def stream_chat_completion(
    messages: list[dict],
    *,
    model: str,
    max_tokens: int = 512,
    base_url: str = "",
) -> AsyncIterator[str]:
    endpoint = _endpoint(base_url)
    url = f"{endpoint.base_url.rstrip('/')}{_chat_path(endpoint)}"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,
    }
    async with (
        pinned_async_client(endpoint, timeout=300.0) as client,
        client.stream("POST", url, json=payload) as resp,
    ):
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line or not line.startswith("data:"):
                continue
            chunk = line[5:].strip()
            if chunk == "[DONE]":
                break
            try:
                data = json.loads(chunk)
            except json.JSONDecodeError:
                continue
            delta = data.get("choices", [{}])[0].get("delta", {})
            content = delta.get("content")
            if content:
                yield content
