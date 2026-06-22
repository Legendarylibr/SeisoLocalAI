"""External LLM provider routing — local vLLM only."""

from __future__ import annotations

import logging
from typing import Any

from forge.security.http_client import pinned_post
from forge.security.url_policy import resolve_pinned_endpoint

logger = logging.getLogger(__name__)

LOCAL_PROVIDER_TYPES = frozenset({"vllm"})


async def chat_completion(
    provider_type: str,
    config: dict[str, Any],
    messages: list[dict],
    *,
    max_tokens: int = 512,
    stream: bool = False,
) -> str:
    ptype = provider_type.lower()
    if ptype == "vllm":
        return await _vllm_compatible(config, messages, max_tokens=max_tokens)
    raise ValueError(f"Unsupported provider: {provider_type}")


async def _vllm_compatible(config: dict, messages: list[dict], max_tokens: int) -> str:
    endpoint = resolve_pinned_endpoint(config.get("base_url", ""), provider_type="vllm")
    model = config.get("model", "default")
    api_key = config.get("api_key", "")

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    path = "/chat/completions"
    if not endpoint.base_url.rstrip("/").endswith("/v1"):
        path = "/v1/chat/completions"

    payload = {"model": model, "messages": messages, "max_tokens": max_tokens}
    resp = await pinned_post(endpoint, path, headers=headers, json=payload)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def mask_config(config: dict) -> dict:
    """Return config safe for API responses (keys redacted)."""
    out = dict(config)
    if "api_key" in out and out["api_key"]:
        out["api_key"] = "***"
    return out
