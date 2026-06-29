from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi import HTTPException

from forge.services.chat_messages import (
    build_trusted_messages,
    prepare_chat_context,
    trim_messages_to_context,
)
from forge.services.model_prompts import (
    chat_system_prompt,
    is_reasoning_prone_model,
    model_switch_system_prompt,
)


class FakeChatDb:
    def __init__(
        self,
        messages: list[dict] | None = None,
        *,
        thread_model_id: str | None = None,
    ) -> None:
        self.messages = messages or []
        self.added: list[tuple[str, str, str]] = []
        self.thread_model_id = thread_model_id
        self.updated_models: list[tuple[str, str | None]] = []

    async def get_messages(self, _thread_id: str) -> list[dict]:
        return self.messages

    async def get_thread_with_messages(
        self, _thread_id: str, _user_id: str
    ) -> tuple[dict | None, list[dict]]:
        thread = await self.get_thread_for_user(_thread_id, _user_id)
        if thread is None:
            return None, []
        return thread, list(self.messages)

    async def add_message(
        self,
        thread_id: str,
        role: str,
        content: str,
        metadata: dict | None = None,
        *,
        model_id: str | None = None,
    ) -> dict:
        self.added.append((thread_id, role, content))
        self.messages.append({"role": role, "content": content})
        if model_id is not None:
            self.updated_models.append((thread_id, model_id))
            self.thread_model_id = model_id
        return {
            "id": "msg-new",
            "thread_id": thread_id,
            "role": role,
            "content": content,
        }

    async def get_thread_for_user(self, _thread_id: str, _user_id: str) -> dict | None:
        return {"id": "thread-1", "model_id": self.thread_model_id}

    async def update_thread_model(self, thread_id: str, model_id: str | None) -> None:
        self.updated_models.append((thread_id, model_id))
        self.thread_model_id = model_id


@pytest.mark.asyncio
async def test_build_trusted_messages_uses_server_history_and_latest_user_turn():
    db = FakeChatDb(
        [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
    )

    messages, user_content = await build_trusted_messages(
        db,
        thread_id="thread-1",
        client_messages=[{"role": "user", "content": "next"}],
        user_id="user-1",
        model_id="Qwen/Qwen3.6-4B",
    )

    assert user_content == "next"
    assert messages[0]["role"] == "system"
    assert "tool" in messages[0]["content"].lower()
    assert messages[-3:] == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "next"},
    ]
    assert db.added == [("thread-1", "user", "next")]


@pytest.mark.asyncio
async def test_build_trusted_messages_rejects_client_assistant_history():
    db = FakeChatDb()

    with pytest.raises(HTTPException, match="Assistant history"):
        await build_trusted_messages(
            db,
            thread_id="thread-1",
            client_messages=[
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
                {"role": "user", "content": "next"},
            ],
        )


@pytest.mark.asyncio
async def test_build_trusted_messages_model_switch_adds_bridge():
    db = FakeChatDb(
        [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ],
        thread_model_id="meta-llama/Llama-4-Scout-17B-16E-Instruct",
    )

    messages, _ = await build_trusted_messages(
        db,
        thread_id="thread-1",
        client_messages=[{"role": "user", "content": "continue"}],
        user_id="user-1",
        model_id="Qwen/Qwen3.6-4B",
    )

    system_messages = [m for m in messages if m["role"] == "system"]
    assert len(system_messages) == 2
    assert "switching" in system_messages[1]["content"].lower()
    assert db.updated_models == [("thread-1", "Qwen/Qwen3.6-4B")]


def test_trim_messages_to_context_keeps_recent_turns_under_budget():
    messages = [
        {"role": "user", "content": "old " * 200},
        {"role": "assistant", "content": "older " * 200},
        {"role": "user", "content": "recent question"},
        {"role": "assistant", "content": "recent answer"},
        {"role": "user", "content": "latest"},
    ]

    trimmed = trim_messages_to_context(messages, budget=120)

    assert trimmed[-1] == {"role": "user", "content": "latest"}
    assert {"role": "user", "content": "old " * 200} not in trimmed
    assert sum(len(m["content"]) + len(m["role"]) + 16 for m in trimmed) <= 120


def test_trim_messages_to_context_truncates_oversized_latest_turn():
    trimmed = trim_messages_to_context(
        [{"role": "user", "content": "x" * 500 + "important ending"}],
        budget=120,
    )

    assert len(trimmed) == 1
    assert trimmed[0]["content"].startswith("[...older content omitted...]")
    assert trimmed[0]["content"].endswith("important ending")
    assert len(trimmed[0]["content"]) + len(trimmed[0]["role"]) + 16 <= 120


def test_trim_messages_to_context_decays_older_messages_by_time():
    now = datetime.now(timezone.utc)
    old_ts = (now - timedelta(hours=6)).isoformat()
    recent_ts = (now - timedelta(minutes=5)).isoformat()
    messages = [
        {"role": "user", "content": "ancient " * 40, "created_at": old_ts},
        {"role": "assistant", "content": "old reply " * 40, "created_at": old_ts},
        {"role": "user", "content": "recent " * 10, "created_at": recent_ts},
        {"role": "user", "content": "latest"},
    ]

    trimmed = trim_messages_to_context(messages, budget=400)

    assert trimmed[-1]["content"] == "latest"
    assert trimmed[0]["content"].startswith("[...older content omitted...]")
    assert len(trimmed[0]["content"]) < len("ancient " * 40)


def test_prepare_chat_context_includes_knowledge_block():
    messages = prepare_chat_context(
        [{"role": "user", "content": "What does the doc say?"}],
        model_key="Qwen/Qwen3.6-4B",
        tools_enabled=False,
        knowledge_context="Reference: Seiso runs locally.",
    )
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "system"
    assert "Reference: Seiso runs locally." in messages[1]["content"]


def test_prepare_chat_context_skips_system_prompt_when_tools_enabled():
    messages = prepare_chat_context(
        [{"role": "user", "content": "hi"}],
        model_key="Qwen/Qwen3.6-4B",
        tools_enabled=True,
    )
    assert messages == [{"role": "user", "content": "hi"}]


def test_chat_system_prompt_includes_thinking_process_hint_for_reasoning_models():
    prompt = chat_system_prompt("Qwen/Qwen3.5-4B", tools_enabled=False)
    assert prompt
    assert "thinking process" in prompt.lower()
    assert "tool" in prompt.lower()
    assert is_reasoning_prone_model("Qwen/Qwen3.5-4B")


def test_chat_system_prompt_includes_coding_guidance_for_all_models():
    for model_key in ("Qwen/Qwen2.5-Coder-7B", "meta-llama/Llama-3.1-8B"):
        prompt = chat_system_prompt(model_key, tools_enabled=False)
        assert prompt
        assert "fenced blocks" in prompt.lower()
        assert "helpful assistant" in prompt.lower()


@pytest.mark.parametrize(
    "model_key",
    [
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "Qwen/Qwen3.6-27B",
        "openai/gpt-oss-20b",
        "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        "google/gemma-3-12b-it",
        "microsoft/Phi-4-mini-instruct",
    ],
)
def test_chat_system_prompt_includes_no_reasoning_hint_for_all_families(model_key: str):
    prompt = chat_system_prompt(model_key, tools_enabled=False)
    assert prompt
    assert "chain-of-thought" in prompt.lower() or "thinking process" in prompt.lower()
    assert "final answer" not in prompt.lower()
    assert "final response" not in prompt.lower()
    assert "meta commentary" in prompt.lower()


def test_model_switch_system_prompt_mentions_models():
    prompt = model_switch_system_prompt(
        "meta-llama/Llama-4-Scout-17B-16E-Instruct", "Qwen/Qwen3.6-4B"
    )
    assert "Llama 4 Scout" in prompt
    assert "Qwen3.6 4B" in prompt
