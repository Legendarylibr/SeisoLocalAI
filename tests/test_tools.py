"""Tests for tool registry."""

import json

from forge.tools.registry import ToolRegistry, ToolSpec, parse_tool_calls


def test_parse_tool_calls():
    text = (
        'Hello <tool_call>{"name": "web_search", "arguments": {"query": "test"}}</tool_call> done'
    )
    calls = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0]["name"] == "web_search"


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
