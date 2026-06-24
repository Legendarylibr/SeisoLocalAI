"""Tests for Nemotron-Orchestrator-8B routing."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from seiso.model_router.catalog import SpecialistCatalog, SpecialistRoute
from seiso.model_router.config import RouterSettings
from seiso.model_router.nemotron import (
    build_answer_tools,
    orchestrator_alias_for,
    parse_tool_calls,
    problem_from_messages,
    select_route_via_nemotron,
    selection_from_tool_calls,
)


def _sample_catalog() -> SpecialistCatalog:
    return SpecialistCatalog(
        routes=[
            SpecialistRoute(
                "general",
                "seiso-general",
                "http://localhost:8001",
                domain_hints=("general", "chat"),
                orchestrator_alias="seiso-general-1",
                metadata={"description": "General chat"},
            ),
            SpecialistRoute(
                "code",
                "seiso-code",
                "http://localhost:8002",
                domain_hints=("code",),
                orchestrator_alias="seiso-code-1",
                metadata={"description": "Code specialist"},
            ),
        ]
    )


def test_orchestrator_alias_default():
    route = SpecialistRoute("reasoning", "seiso-reasoning", "http://localhost:8003")
    assert orchestrator_alias_for(route) == "seiso-reasoning-1"


def test_build_answer_tools_includes_aliases():
    catalog = _sample_catalog()
    tools = build_answer_tools(catalog)
    assert tools[0]["function"]["name"] == "answer"
    enum = tools[0]["function"]["parameters"]["properties"]["model"]["enum"]
    assert "seiso-general-1" in enum
    assert "seiso-code-1" in enum


def test_problem_from_messages_flattens_history():
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "write python"},
    ]
    problem = problem_from_messages(messages)
    assert "hello" in problem
    assert "write python" in problem


def test_parse_tool_calls_openai_format():
    payload = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "answer",
                                "arguments": json.dumps({"model": "seiso-code-1"}),
                            }
                        }
                    ]
                }
            }
        ]
    }
    calls = parse_tool_calls(payload)
    assert calls[0]["name"] == "answer"
    assert calls[0]["arguments"]["model"] == "seiso-code-1"


def test_selection_from_tool_calls_maps_alias():
    catalog = _sample_catalog()
    selection = selection_from_tool_calls(
        catalog,
        [{"name": "answer", "arguments": {"model": "seiso-code-1"}}],
        fallback_route_id="general",
    )
    assert selection.route.route_id == "code"
    assert selection.orchestrator_alias == "seiso-code-1"


def test_selection_from_tool_calls_falls_back():
    catalog = _sample_catalog()
    selection = selection_from_tool_calls(catalog, [], fallback_route_id="general")
    assert selection.route.route_id == "general"
    assert "fallback" in selection.reasoning


@pytest.mark.asyncio
async def test_select_route_via_nemotron_http():

    catalog = _sample_catalog()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = ""
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "answer",
                                "arguments": '{"model": "seiso-code-1"}',
                            }
                        }
                    ]
                }
            }
        ]
    }
    client = AsyncMock()
    client.post = AsyncMock(return_value=mock_response)

    selection = await select_route_via_nemotron(
        client,
        orchestrator_url="http://orchestrator:8000",
        orchestrator_model="seiso-orchestrator",
        catalog=catalog,
        messages=[{"role": "user", "content": "def foo(): pass"}],
        fallback_route_id="general",
    )
    assert selection.route.route_id == "code"
    client.post.assert_awaited_once()
    call_kwargs = client.post.await_args.kwargs
    assert "tools" in call_kwargs["json"]


def test_nemotron_orchestrator_enabled_requires_vllm_sleep():
    settings = RouterSettings(
        routing_mode="nemotron",
        inference_backend="vllm",
        vllm_sleep_mode=True,
        orchestrator_url="http://vllm-orchestrator:8000",
    )
    assert settings.nemotron_orchestrator_enabled() is True


def test_nemotron_orchestrator_disabled_for_llamacpp():
    settings = RouterSettings.model_construct(
        routing_mode="nemotron",
        inference_backend="llamacpp",
        vllm_sleep_mode=False,
        orchestrator_url="http://vllm-orchestrator:8000",
    )
    assert settings.nemotron_orchestrator_enabled() is False


def test_nemotron_orchestrator_disabled_without_sleep_mode():
    settings = RouterSettings.model_construct(
        routing_mode="nemotron",
        inference_backend="vllm",
        vllm_sleep_mode=False,
        orchestrator_url="http://vllm-orchestrator:8000",
    )
    assert settings.nemotron_orchestrator_enabled() is False


def test_nemotron_config_rejected_without_vllm_sleep():
    with pytest.raises(ValueError, match="vLLM sleep mode"):
        RouterSettings(
            routing_mode="nemotron",
            inference_backend="llamacpp",
            orchestrator_url="http://vllm-orchestrator:8000",
        )


def test_nemotron_config_rejected_without_sleep_flag():
    with pytest.raises(ValueError, match="vllm_sleep_mode"):
        RouterSettings(
            routing_mode="nemotron",
            inference_backend="vllm",
            vllm_sleep_mode=False,
            orchestrator_url="http://vllm-orchestrator:8000",
        )
