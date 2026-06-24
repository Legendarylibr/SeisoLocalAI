"""LiteLLM execution gateway — Nemotron-selected routes to local and cloud backends."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import AsyncIterator
from typing import Any

from seiso.model_router.catalog import SpecialistCatalog, SpecialistRoute

logger = logging.getLogger(__name__)


def _openai_api_base(url: str) -> str:
    base = url.rstrip("/")
    if base.endswith("/v1"):
        return base
    return f"{base}/v1"


def _resolve_api_key(route: SpecialistRoute) -> str | None:
    if not route.api_key_env:
        return None
    key = os.environ.get(route.api_key_env, "").strip()
    return key or None


def build_litellm_params(
    route: SpecialistRoute,
    *,
    llamaswap_url: str = "",
) -> dict[str, Any]:
    """Map a catalog route to LiteLLM Router litellm_params."""
    params: dict[str, Any] = {}

    if route.backend_type == "vllm":
        if llamaswap_url:
            params["api_base"] = _openai_api_base(llamaswap_url)
            params["model"] = route.litellm_model or f"hosted_vllm/{route.llamaswap_model}"
        else:
            params["api_base"] = _openai_api_base(route.backend_url)
            params["model"] = route.litellm_model or f"hosted_vllm/{route.openai_model_name}"
    elif route.backend_type == "cloud_vllm":
        params["api_base"] = _openai_api_base(route.backend_url)
        params["model"] = route.litellm_model or f"hosted_vllm/{route.openai_model_name}"
    elif route.backend_type == "cloud_api":
        params["model"] = route.litellm_model
        if route.backend_url:
            params["api_base"] = _openai_api_base(route.backend_url)
    else:
        raise ValueError(f"route {route.route_id!r} is not LiteLLM-routable")

    api_key = _resolve_api_key(route)
    if api_key:
        params["api_key"] = api_key

    return params


def build_litellm_model_list(
    catalog: SpecialistCatalog,
    *,
    llamaswap_url: str = "",
) -> list[dict[str, Any]]:
    """Build LiteLLM Router model_list for all LiteLLM-backed specialist routes."""
    entries: list[dict[str, Any]] = []
    for route in catalog.litellm_routes():
        entries.append(
            {
                "model_name": route.llamaswap_model,
                "litellm_params": build_litellm_params(route, llamaswap_url=llamaswap_url),
            }
        )
    if not entries:
        raise ValueError("LiteLLM gateway requires at least one LiteLLM-routable specialist")
    return entries


def litellm_response_to_openai(response: Any) -> dict[str, Any]:
    if hasattr(response, "model_dump"):
        return response.model_dump()
    if isinstance(response, dict):
        return response
    raise TypeError(f"Unexpected LiteLLM response type: {type(response)!r}")


class LitellmGateway:
    """Execute completions via LiteLLM after Nemotron/heuristic route selection."""

    def __init__(
        self,
        catalog: SpecialistCatalog,
        *,
        llamaswap_url: str = "",
        routing_strategy: str = "simple-shuffle",
        request_timeout_sec: float = 300.0,
    ) -> None:
        from litellm import Router

        self._timeout = request_timeout_sec
        model_list = build_litellm_model_list(catalog, llamaswap_url=llamaswap_url)
        self._router = Router(
            model_list=model_list,
            routing_strategy=routing_strategy,
            num_retries=2,
            timeout=request_timeout_sec,
        )
        logger.info(
            "LiteLLM gateway: %d routes (strategy=%s)",
            len(model_list),
            routing_strategy,
        )

    @staticmethod
    def _model_name(route: SpecialistRoute) -> str:
        return route.llamaswap_model

    async def chat_completion(
        self,
        route: SpecialistRoute,
        *,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float,
    ) -> dict[str, Any]:
        response = await self._router.acompletion(
            model=self._model_name(route),
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False,
            timeout=self._timeout,
        )
        return litellm_response_to_openai(response)

    async def stream_chat_completion(
        self,
        route: SpecialistRoute,
        *,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float,
    ) -> AsyncIterator[str]:
        response = await self._router.acompletion(
            model=self._model_name(route),
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            timeout=self._timeout,
        )
        async for chunk in response:
            payload = litellm_response_to_openai(chunk)
            yield f"data: {json.dumps(payload)}\n\n"
        yield "data: [DONE]\n\n"
