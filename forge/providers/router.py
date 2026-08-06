"""External chat-server routing — local and optional remote multi-GPU.

Provider-agnostic: any server that speaks the standard chat-completions
HTTP wire protocol (``/v1/chat/completions``) works. Canonical types:

* ``local_chat``  — loopback multi-GPU / local chat server (alias: ``vllm``)
* ``remote_chat`` — remote HTTPS multi-GPU chat server (alias: ``vllm_cloud``)

No vendor account is required; the wire protocol is shared by vLLM, SGLang,
Ollama-compatible gateways, RunPod pods, and many others.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from forge.security.http_client import pinned_async_client, pinned_post
from forge.security.url_policy import resolve_pinned_endpoint
from seiso.env import env_bool

logger = logging.getLogger(__name__)

# Canonical provider types (preferred in UI and new configs).
PROVIDER_LOCAL_CHAT = "local_chat"
PROVIDER_REMOTE_CHAT = "remote_chat"

# Backward-compatible aliases (existing installs / docs).
_LOCAL_ALIASES = frozenset({PROVIDER_LOCAL_CHAT, "vllm"})
_REMOTE_ALIASES = frozenset({PROVIDER_REMOTE_CHAT, "vllm_cloud"})

# Public sets used by routes (includes aliases).
LOCAL_PROVIDER_TYPES = _LOCAL_ALIASES
CLOUD_MULTIGPU_PROVIDER_TYPES = _REMOTE_ALIASES
CHAT_PROVIDER_TYPES = LOCAL_PROVIDER_TYPES | CLOUD_MULTIGPU_PROVIDER_TYPES


def normalize_provider_type(provider_type: str) -> str:
    """Map aliases to canonical type names."""
    ptype = (provider_type or "").strip().lower()
    if ptype in _LOCAL_ALIASES:
        return PROVIDER_LOCAL_CHAT
    if ptype in _REMOTE_ALIASES:
        return PROVIDER_REMOTE_CHAT
    return ptype


def cloud_multigpu_enabled() -> bool:
    """Opt-in gate for remote multi-GPU chat servers."""
    return env_bool("SEISO_ALLOW_CLOUD_MULTIGPU", False) or env_bool(
        "SEISO_ALLOW_CLOUD_PROVIDERS", False
    )


def is_chat_provider_type(provider_type: str) -> bool:
    ptype = (provider_type or "").lower()
    if ptype in LOCAL_PROVIDER_TYPES:
        return True
    if ptype in CLOUD_MULTIGPU_PROVIDER_TYPES:
        return cloud_multigpu_enabled()
    return False


def allowed_chat_provider_types() -> frozenset[str]:
    """Types accepted on create/list (canonical + aliases)."""
    allowed = set(LOCAL_PROVIDER_TYPES)
    if cloud_multigpu_enabled():
        allowed |= CLOUD_MULTIGPU_PROVIDER_TYPES
    return frozenset(allowed)


def preferred_chat_provider_types() -> list[str]:
    """Canonical types to advertise in API docs / UI defaults."""
    out = [PROVIDER_LOCAL_CHAT]
    if cloud_multigpu_enabled():
        out.append(PROVIDER_REMOTE_CHAT)
    return out


async def chat_completion(
    provider_type: str,
    config: dict[str, Any],
    messages: list[dict],
    *,
    max_tokens: int = 512,
    temperature: float | None = None,
    stream: bool = False,
) -> str:
    """Non-streaming chat completion against a standard chat-completions server."""
    ptype = normalize_provider_type(provider_type)
    if ptype not in {PROVIDER_LOCAL_CHAT, PROVIDER_REMOTE_CHAT}:
        raise ValueError(f"Unsupported provider: {provider_type}")
    if ptype == PROVIDER_REMOTE_CHAT and not cloud_multigpu_enabled():
        raise ValueError(
            "Remote multi-GPU chat servers are disabled. "
            "Set SEISO_ALLOW_CLOUD_MULTIGPU=true to enable."
        )
    if stream:
        parts: list[str] = []
        async for token in stream_chat_completion(
            provider_type,
            config,
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
        ):
            parts.append(token)
        return "".join(parts)
    return await _chat_completions_request(
        ptype, config, messages, max_tokens=max_tokens, temperature=temperature
    )


async def stream_chat_completion(
    provider_type: str,
    config: dict[str, Any],
    messages: list[dict],
    *,
    max_tokens: int = 512,
    temperature: float | None = None,
) -> AsyncIterator[str]:
    """Stream text deltas from a standard chat-completions server (SSE)."""
    ptype = normalize_provider_type(provider_type)
    if ptype not in {PROVIDER_LOCAL_CHAT, PROVIDER_REMOTE_CHAT}:
        raise ValueError(f"Unsupported provider: {provider_type}")
    if ptype == PROVIDER_REMOTE_CHAT and not cloud_multigpu_enabled():
        raise ValueError(
            "Remote multi-GPU chat servers are disabled. "
            "Set SEISO_ALLOW_CLOUD_MULTIGPU=true to enable."
        )

    endpoint = resolve_pinned_endpoint(config.get("base_url", ""), provider_type=ptype)
    model = config.get("model") or config.get("compat_model_id") or "default"
    api_key = config.get("api_key", "")

    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    path = "/chat/completions"
    if not endpoint.base_url.rstrip("/").endswith("/v1"):
        path = "/v1/chat/completions"

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,
    }
    if temperature is not None:
        payload["temperature"] = temperature

    timeout = float(config.get("timeout_s") or 600)
    url = f"{endpoint.base_url.rstrip('/')}{path}"
    async with (
        pinned_async_client(endpoint, timeout=timeout) as client,
        client.stream("POST", url, headers=headers, json=payload) as resp,
    ):
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line:
                continue
            if line.startswith("data:"):
                data = line[5:].strip()
            else:
                continue
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            if isinstance(chunk, dict) and chunk.get("error"):
                err = chunk["error"]
                msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
                raise RuntimeError(msg)
            choices = chunk.get("choices") if isinstance(chunk, dict) else None
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            content = delta.get("content")
            if content:
                yield str(content)


async def _chat_completions_request(
    provider_type: str,
    config: dict,
    messages: list[dict],
    *,
    max_tokens: int,
    temperature: float | None = None,
) -> str:
    endpoint = resolve_pinned_endpoint(config.get("base_url", ""), provider_type=provider_type)
    model = config.get("model") or config.get("compat_model_id") or "default"
    api_key = config.get("api_key", "")

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    path = "/chat/completions"
    if not endpoint.base_url.rstrip("/").endswith("/v1"):
        path = "/v1/chat/completions"

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if temperature is not None:
        payload["temperature"] = temperature

    timeout = float(config.get("timeout_s") or 600)
    resp = await pinned_post(endpoint, path, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def mask_config(config: dict) -> dict:
    """Return config safe for API responses (keys redacted)."""
    out = dict(config)
    if "api_key" in out and out["api_key"]:
        out["api_key"] = "***"
    return out


def normalize_remote_chat_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate/normalize remote multi-GPU chat server config fields."""
    out = dict(config)
    out["deployment_kind"] = "multi_gpu_remote"
    if "tensor_parallel_size" in out and out["tensor_parallel_size"] is not None:
        tp = int(out["tensor_parallel_size"])
        if tp < 1 or tp > 1024:
            raise ValueError("tensor_parallel_size must be between 1 and 1024")
        out["tensor_parallel_size"] = tp
    if "gpu_count" in out and out["gpu_count"] is not None:
        gc = int(out["gpu_count"])
        if gc < 1 or gc > 1024:
            raise ValueError("gpu_count must be between 1 and 1024")
        out["gpu_count"] = gc
    # Optional metadata: which hoster/engine (runpod, sglang, vllm, …) — not a wire type.
    engine = str(out.get("engine") or out.get("backend") or "").strip()
    if engine:
        out["engine"] = engine.lower()[:64]
    hoster = str(
        out.get("cloud_provider") or out.get("hoster") or out.get("provider") or ""
    ).strip()
    if hoster:
        out["hoster"] = hoster.lower()[:64]
        out["cloud_provider"] = out["hoster"]  # keep legacy key for older UI
    region = str(out.get("region") or "").strip()
    if region:
        out["region"] = region[:128]
    model = str(out.get("model") or out.get("compat_model_id") or "").strip()
    if model:
        out["model"] = model
    return out


# Back-compat name used by older imports/tests.
normalize_cloud_multigpu_config = normalize_remote_chat_config
