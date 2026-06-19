from __future__ import annotations

import pytest
from fastapi import HTTPException

from forge.services.chat_messages import build_trusted_messages, trim_messages_to_context


class FakeChatDb:
    def __init__(self, messages: list[dict] | None = None) -> None:
        self.messages = messages or []
        self.added: list[tuple[str, str, str]] = []

    async def get_messages(self, _thread_id: str) -> list[dict]:
        return self.messages

    async def add_message(self, thread_id: str, role: str, content: str) -> dict:
        self.added.append((thread_id, role, content))
        self.messages.append({"role": role, "content": content})
        return {"id": "msg-new", "thread_id": thread_id, "role": role, "content": content}


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
    )

    assert user_content == "next"
    assert messages == [
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
