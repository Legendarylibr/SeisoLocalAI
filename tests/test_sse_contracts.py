from __future__ import annotations

import pytest

from tests.conftest import user_path


def _sse_events(raw: bytes) -> list[tuple[str, str]]:
    events: list[tuple[str, str]] = []
    for block in raw.decode().replace("\r\n", "\n").split("\n\n"):
        if not block.strip():
            continue
        event = "message"
        data: list[str] = []
        for line in block.splitlines():
            if line.startswith("event:"):
                event = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data.append(line.split(":", 1)[1].lstrip())
        events.append((event, "\n".join(data)))
    return events


@pytest.mark.asyncio
async def test_chat_stream_sends_token_message_done(app, auth_client, monkeypatch):
    client, _token, headers, data_dir = auth_client
    from forge.api.deps import get_db

    db = get_db()
    user = await db.get_user_by_display_name("Admin")
    model_path = user_path(data_dir, user["id"], "models", "model.gguf")
    model_path.write_text("fake")
    model = await db.add_model(
        user_id=user["id"], name="Local", path=str(model_path), format="gguf"
    )

    async def fake_stream(_self, _payload):
        yield "hello "
        yield "world"

    monkeypatch.setattr("forge.orchestrators.inference.LocalInferenceRunner.stream", fake_stream)

    async with client.stream(
        "POST",
        "/api/inference/chat",
        headers=headers,
        json={
            "model_id": model["id"],
            "inference_backend": "llamacpp",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    ) as res:
        assert res.status_code == 200
        events = _sse_events(await res.aread())

    token_text = "".join(data for event, data in events if event == "token")
    assert token_text == "hello world"
    assert ("message", "hello world") in events
    assert any(event == "done" for event, _data in events)
