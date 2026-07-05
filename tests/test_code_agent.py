from __future__ import annotations

from pathlib import Path

import pytest

from forge.services.code_agent import (
    build_code_agent_system_prompt,
    build_code_tool_registry,
    infer_model_family,
    max_concurrent_code_agents,
    resolve_code_agent_model_id,
)
from forge.services.code_workspace import run_terminal


def test_infer_model_family_detects_gemma_and_qwen():
    assert infer_model_family("google/gemma-3-4b-it-Q4_K_M.gguf") == "gemma"
    assert infer_model_family("Qwen/Qwen2.5-Coder-7B-GGUF") == "qwen"


def test_build_code_agent_system_prompt_mentions_patch_tools_for_gemma():
    prompt = build_code_agent_system_prompt(
        model_key="gemma-3",
        enabled_tools=["repo.read", "files.write"],
    )
    assert "patch_file" in prompt
    assert "gemma" in prompt.lower() or "Gemma" in prompt


def test_build_code_tool_registry_respects_enabled_tools(tmp_path: Path):
    class Settings:
        data_dir = tmp_path

    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.txt").write_text("hello", encoding="utf-8")

    reg = build_code_tool_registry(root, Settings(), ["repo.read"])
    assert "read_file" in reg.tools
    assert "write_file" not in reg.tools

    result = reg.execute("read_file", {"path": "a.txt"})
    assert "hello" in result
    assert "[TOOL_DATA source=read_file]" in result


def test_build_code_tool_registry_includes_terminal_read(tmp_path: Path):
    class Settings:
        data_dir = tmp_path
        web_search_enabled = False

    root = tmp_path / "repo"
    root.mkdir()

    reg = build_code_tool_registry(root, Settings(), ["terminal.read"])
    assert "read_terminal_log" in reg.tools


def test_recent_terminal_output_after_run(tmp_path: Path, monkeypatch):
    from forge.services.code_workspace import recent_terminal_output

    root = tmp_path / "repo"
    root.mkdir()

    def fake_run(*args, **kwargs):
        class Result:
            returncode = 0
            stdout = "ok\n"
            stderr = ""

        return Result()

    monkeypatch.setattr("forge.services.code_workspace.subprocess.run", fake_run)
    run_terminal(root, "echo ok")
    log = recent_terminal_output(root)
    assert log["count"] == 1
    assert log["entries"][0]["output"] == "ok\n"


def test_max_concurrent_code_agents_is_at_least_one():
    assert max_concurrent_code_agents({"gpus": [], "ram_gb": 8}) >= 1


@pytest.mark.asyncio
async def test_resolve_code_agent_model_id_rejects_blocked_model(monkeypatch):
    class FakeDb:
        async def get_model(self, model_id, user_id):
            return {
                "id": model_id,
                "name": "Huge",
                "path": "/tmp/huge.gguf",
                "format": "gguf",
                "metadata_json": "{}",
                "size_bytes": 1,
                "source": "hf:test",
            }

    async def fake_get_option(db, user_id, model_id, **kwargs):
        return {
            "id": model_id,
            "hardware_fit": "blocked",
            "hardware_fit_reason": "Too large for this machine",
        }

    monkeypatch.setattr(
        "forge.services.code_agent.get_inference_option",
        fake_get_option,
    )

    class Settings:
        model_router_enabled = False

    with pytest.raises(ValueError, match="Too large"):
        await resolve_code_agent_model_id(FakeDb(), "user", Settings(), "huge-model")
