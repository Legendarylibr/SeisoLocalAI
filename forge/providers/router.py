"""External LLM provider routing — OpenAI, Anthropic, Ollama, vLLM."""

from __future__ import annotations

import logging
from typing import Any

from forge.security.http_client import pinned_post
from forge.security.url_policy import resolve_pinned_endpoint

logger = logging.getLogger(__name__)


async def chat_completion(
    provider_type: str,
    config: dict[str, Any],
    messages: list[dict],
    *,
    max_tokens: int = 512,
    stream: bool = False,
) -> str:
    ptype = provider_type.lower()
    if ptype in ("openai", "vllm", "ollama"):
        return await _openai_compatible(config, messages, max_tokens=max_tokens, provider_type=ptype)
    if ptype == "anthropic":
        return await _anthropic(config, messages, max_tokens=max_tokens)
    raise ValueError(f"Unsupported provider: {provider_type}")


async def _openai_compatible(
    config: dict,
    messages: list[dict],
    max_tokens: int,
    *,
    provider_type: str = "openai",
) -> str:
    if provider_type == "ollama":
        from forge.providers.ollama import chat_completion as ollama_chat

        return await ollama_chat(
            messages,
            model=config.get("model", "llama3.2"),
            max_tokens=max_tokens,
            base_url=config.get("base_url", ""),
        )

    endpoint = resolve_pinned_endpoint(config.get("base_url", ""), provider_type=provider_type)
    model = config.get("model", "gpt-4o-mini")
    api_key = config.get("api_key", "")

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    path = "/chat/completions"
    if provider_type == "vllm" and not endpoint.base_url.rstrip("/").endswith("/v1"):
        path = "/v1/chat/completions"

    payload = {"model": model, "messages": messages, "max_tokens": max_tokens}
    resp = await pinned_post(endpoint, path, headers=headers, json=payload)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


async def _anthropic(config: dict, messages: list[dict], max_tokens: int) -> str:
    api_key = config.get("api_key", "")
    model = config.get("model", "claude-3-5-sonnet-20241022")
    endpoint = resolve_pinned_endpoint(config.get("base_url", ""), provider_type="anthropic")

    system = ""
    api_messages = []
    for m in messages:
        if m.get("role") == "system":
            system = m.get("content", "")
        else:
            api_messages.append({"role": m["role"], "content": m.get("content", "")})

    payload: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": api_messages,
    }
    if system:
        payload["system"] = system

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    resp = await pinned_post(endpoint, "/v1/messages", headers=headers, json=payload)
    resp.raise_for_status()
    data = resp.json()
    parts = data.get("content", [])
    return "".join(p.get("text", "") for p in parts if p.get("type") == "text")


def mask_config(config: dict) -> dict:
    """Return config safe for API responses (keys redacted)."""
    out = dict(config)
    if "api_key" in out and out["api_key"]:
        out["api_key"] = "***"
    return out
