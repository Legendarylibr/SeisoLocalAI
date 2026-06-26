"""Tests for tool registry."""

import json

from forge.tools.registry import (
    ToolRegistry,
    ToolSpec,
    parse_tool_calls,
    resolve_tool_call_format,
    tools_system_prompt,
)


def test_parse_xml_function_tool_calls():
    text = (
        "<tool_call><function=web_search>"
        "<parameter=query>latest AI news</parameter>"
        "</function></tool_call>"
    )
    calls = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0]["name"] == "web_search"
    assert calls[0]["arguments"]["query"] == "latest AI news"


def test_parse_tool_calls():
    text = (
        'Hello <tool_call>{"name": "web_search", "arguments": {"query": "test"}}</tool_call> done'
    )
    calls = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0]["name"] == "web_search"


def test_tools_system_prompt_uses_xml_format_for_qwen_family():
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="web_search",
            description="Search the web",
            parameters={"type": "object"},
            handler=lambda: "",
        )
    )
    prompt = tools_system_prompt(registry, model_key="Qwen/Qwen3-4B")
    assert "<function=TOOL>" in prompt
    assert "JSON blocks" not in prompt


def test_tools_system_prompt_includes_generic_coding_guidance():
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="web_search",
            description="Search the web",
            parameters={"type": "object"},
            handler=lambda: "",
        )
    )
    prompt = tools_system_prompt(registry, model_key="mistralai/Devstral-Small-2505")
    assert "fenced blocks" in prompt.lower()
    assert "execute_code" not in prompt
    assert "write_artifact" not in prompt


def test_tools_system_prompt_uses_json_format_for_llama_family():
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="web_search",
            description="Search the web",
            parameters={"type": "object"},
            handler=lambda: "",
        )
    )
    prompt = tools_system_prompt(registry, model_key="meta-llama/Llama-3.1-8B")
    assert "JSON blocks" in prompt
    assert "<function=TOOL>" not in prompt


def test_parse_tool_calls_prefers_xml_for_qwen_family():
    xml = (
        "<tool_call><function=web_search>"
        "<parameter=query>news</parameter>"
        "</function></tool_call>"
    )
    json_call = (
        '<tool_call>{"name": "web_search", "arguments": {"query": "news"}}</tool_call>'
    )
    calls = parse_tool_calls(xml + json_call, model_key="Qwen/Qwen3-4B")
    assert len(calls) == 1
    assert calls[0]["name"] == "web_search"


def test_parse_mistral_tool_calls_bracket_format():
    text = '[TOOL_CALLS] [{"name": "web_search", "arguments": {"query": "news"}}]'
    calls = parse_tool_calls(text, model_key="mistralai/Mistral-7B")
    assert len(calls) == 1
    assert calls[0]["arguments"]["query"] == "news"


def test_resolve_tool_call_format_by_architecture_not_model_id():
    assert resolve_tool_call_format("org/any-qwen3-variant") == "xml_function"
    assert resolve_tool_call_format("org/any-llama-variant") == "json_tagged"


def test_tool_registry_execute():
    reg = ToolRegistry()
    reg.register(
        ToolSpec(
            name="echo",
            description="echo",
            parameters={
                "type": "object",
                "properties": {"msg": {"type": "string"}},
                "required": ["msg"],
            },
            handler=lambda msg: json.dumps({"echo": msg}),
        )
    )
    out = reg.execute("echo", {"msg": "hi"})
    assert json.loads(out)["echo"] == "hi"
