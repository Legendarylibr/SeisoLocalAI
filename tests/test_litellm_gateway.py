"""Tests for LiteLLM execution gateway."""

from __future__ import annotations

import os
import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from seiso.model_router.catalog import SpecialistCatalog, SpecialistRoute
from seiso.model_router.config import RouterSettings
from seiso.model_router.litellm_gateway import (
    LitellmGateway,
    build_litellm_model_list,
    build_litellm_params,
    litellm_response_to_openai,
)


def _local_catalog() -> SpecialistCatalog:
    return SpecialistCatalog(
        routes=[
            SpecialistRoute(
                "general",
                "seiso-general",
                "http://vllm-general:8000",
                backend_type="vllm",
            ),
            SpecialistRoute(
                "code",
                "seiso-code",
                "http://vllm-code:8000",
                backend_type="vllm",
            ),
        ]
    )


def test_litellm_gateway_enabled_for_vllm_stack():
    settings = RouterSettings(inference_backend="vllm", vllm_sleep_mode=True)
    assert settings.litellm_gateway_enabled() is True


def test_litellm_gateway_disabled_for_llamacpp():
    settings = RouterSettings(inference_backend="llamacpp")
    assert settings.litellm_gateway_enabled() is False


def test_build_litellm_model_list_direct_vllm():
    models = build_litellm_model_list(_local_catalog())
    assert len(models) == 2
    assert models[0]["model_name"] == "seiso-general"
    assert models[0]["litellm_params"]["api_base"] == "http://vllm-general:8000/v1"


def test_build_litellm_params_cloud_api():
    route = SpecialistRoute(
        "cloud",
        "seiso-cloud",
        "",
        backend_type="cloud_api",
        litellm_model="openai/gpt-4o-mini",
        api_key_env="OPENAI_API_KEY",
    )
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
        params = build_litellm_params(route)
    assert params["model"] == "openai/gpt-4o-mini"
    assert params["api_key"] == "sk-test"


def test_build_litellm_params_cloud_vllm():
    route = SpecialistRoute(
        "cloud-code",
        "seiso-cloud-code",
        "https://gpu.example.com",
        backend_type="cloud_vllm",
        litellm_model="hosted_vllm/code-32b",
    )
    params = build_litellm_params(route)
    assert params["api_base"] == "https://gpu.example.com/v1"
    assert params["model"] == "hosted_vllm/code-32b"


def test_cloud_api_requires_litellm_model():
    with pytest.raises(ValueError, match="litellm_model"):
        SpecialistRoute("x", "x", "", backend_type="cloud_api")


@pytest.mark.asyncio
async def test_litellm_gateway_chat_completion():
    mock_router = MagicMock()
    mock_response = MagicMock()
    mock_response.model_dump.return_value = {
        "choices": [{"message": {"content": "answer"}}],
    }
    mock_router.acompletion = AsyncMock(return_value=mock_response)

    litellm_mod = types.ModuleType("litellm")
    litellm_mod.Router = MagicMock(return_value=mock_router)
    with patch.dict(sys.modules, {"litellm": litellm_mod}):
        gateway = LitellmGateway(_local_catalog(), llamaswap_url="http://llama-swap:8080")
        route = _local_catalog().by_id("code")
        out = await gateway.chat_completion(
            route,
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=32,
            temperature=0.5,
        )

    assert out["choices"][0]["message"]["content"] == "answer"
    assert mock_router.acompletion.await_args.kwargs["model"] == "seiso-code"


def test_litellm_response_to_openai_model_dump():
    resp = MagicMock()
    resp.model_dump.return_value = {"choices": [{"message": {"content": "hi"}}]}
    assert litellm_response_to_openai(resp)["choices"][0]["message"]["content"] == "hi"
