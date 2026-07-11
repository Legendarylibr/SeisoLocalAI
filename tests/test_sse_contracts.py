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

    from seiso.inference.streaming import StreamUpdate

    async def fake_stream_updates(_payload):
        yield StreamUpdate(text="hello ", output_tokens=1)
        yield StreamUpdate(
            text="world",
            output_tokens=2,
            metadata={"finish_reason": "stop"},
        )

    from unittest.mock import MagicMock

    from forge.api.deps import get_inference_orchestrator
    from forge.services import hf_connectivity, inference_models

    monkeypatch.setattr(
        inference_models,
        "_installed_backends",
        lambda: {"llamacpp": True, "llamaswap": False, "mlx": False, "torch": False},
    )
    monkeypatch.setattr(
        hf_connectivity,
        "check_inference_runtime",
        lambda: type(
            "R",
            (),
            {"llamacpp": True, "llamaswap": False, "mlx": False, "torch": False},
        )(),
    )
    monkeypatch.setattr(
        "seiso.inference.backends._native_linux_requires_isolated_gguf",
        lambda: False,
    )

    mock_runner = MagicMock()
    mock_runner.stream_updates = fake_stream_updates
    mock_runner.pool.has_active_inference.return_value = False
    get_inference_orchestrator()._runner = mock_runner

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
        assert res.status_code == 200, await res.aread()
        events = _sse_events(await res.aread())

    token_text = "".join(data for event, data in events if event == "token")
    assert token_text == "hello world"
    assert ("message", "hello world") in events
    assert any(event == "done" for event, _data in events)
    stats = [data for event, data in events if event == "stats"]
    assert stats
    assert '"output_tokens": 2' in stats[-1]
    assert '"truncated": false' in stats[-1]


def _install_chat_stream_mocks(monkeypatch, mock_runner):
    from forge.api.deps import get_inference_orchestrator
    from forge.services import hf_connectivity, inference_models

    monkeypatch.setattr(
        inference_models,
        "_installed_backends",
        lambda: {"llamacpp": True, "llamaswap": False, "mlx": False, "torch": False},
    )
    monkeypatch.setattr(
        hf_connectivity,
        "check_inference_runtime",
        lambda: type(
            "R",
            (),
            {"llamacpp": True, "llamaswap": False, "mlx": False, "torch": False},
        )(),
    )
    monkeypatch.setattr(
        "seiso.inference.backends._native_linux_requires_isolated_gguf",
        lambda: False,
    )
    get_inference_orchestrator()._runner = mock_runner


@pytest.mark.asyncio
async def test_chat_stream_auto_continues_on_length_limit(app, auth_client, monkeypatch):
    """When a pass hits max_tokens, server continues once and concatenates tokens."""
    monkeypatch.setenv("SEISO_CHAT_AUTO_CONTINUE_MAX", "2")
    client, _token, headers, data_dir = auth_client
    from forge.api.deps import get_db

    db = get_db()
    user = await db.get_user_by_display_name("Admin")
    model_path = user_path(data_dir, user["id"], "models", "model.gguf")
    model_path.write_text("fake")
    model = await db.add_model(
        user_id=user["id"], name="Local", path=str(model_path), format="gguf"
    )

    from seiso.inference.streaming import StreamUpdate

    calls: list[dict] = []

    async def fake_stream_updates(payload):
        calls.append(payload)
        if len(calls) == 1:
            yield StreamUpdate(
                text="Part one ",
                output_tokens=8,
                metadata={"finish_reason": "length"},
            )
            return
        yield StreamUpdate(
            text="and part two.",
            output_tokens=4,
            metadata={"finish_reason": "stop"},
        )

    from unittest.mock import MagicMock

    mock_runner = MagicMock()
    mock_runner.stream_updates = fake_stream_updates
    mock_runner.pool.has_active_inference.return_value = False
    _install_chat_stream_mocks(monkeypatch, mock_runner)

    async with client.stream(
        "POST",
        "/api/inference/chat",
        headers=headers,
        json={
            "model_id": model["id"],
            "inference_backend": "llamacpp",
            "messages": [{"role": "user", "content": "write a long answer"}],
            "max_tokens": 8,
            "stream": True,
        },
    ) as res:
        assert res.status_code == 200, await res.aread()
        events = _sse_events(await res.aread())

    assert len(calls) == 2
    assert calls[1].get("max_tokens") == 8
    token_text = "".join(data for event, data in events if event == "token")
    assert token_text == "Part one and part two."
    assert ("message", "Part one and part two.") in events
    logs = [data for event, data in events if event == "log"]
    assert any("continuing" in line.lower() for line in logs)
    stats = [data for event, data in events if event == "stats"]
    assert stats
    assert '"auto_continues": 1' in stats[-1]
    assert '"truncated": false' in stats[-1]


@pytest.mark.asyncio
async def test_chat_stream_auto_continue_disabled(app, auth_client, monkeypatch):
    monkeypatch.setenv("SEISO_CHAT_AUTO_CONTINUE_MAX", "0")
    client, _token, headers, data_dir = auth_client
    from forge.api.deps import get_db

    db = get_db()
    user = await db.get_user_by_display_name("Admin")
    model_path = user_path(data_dir, user["id"], "models", "model.gguf")
    model_path.write_text("fake")
    model = await db.add_model(
        user_id=user["id"], name="Local", path=str(model_path), format="gguf"
    )

    from seiso.inference.streaming import StreamUpdate

    calls = 0

    async def fake_stream_updates(_payload):
        nonlocal calls
        calls += 1
        yield StreamUpdate(
            text="cut off mid",
            output_tokens=8,
            metadata={"finish_reason": "length"},
        )

    from unittest.mock import MagicMock

    mock_runner = MagicMock()
    mock_runner.stream_updates = fake_stream_updates
    mock_runner.pool.has_active_inference.return_value = False
    _install_chat_stream_mocks(monkeypatch, mock_runner)

    async with client.stream(
        "POST",
        "/api/inference/chat",
        headers=headers,
        json={
            "model_id": model["id"],
            "inference_backend": "llamacpp",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8,
            "stream": True,
        },
    ) as res:
        assert res.status_code == 200, await res.aread()
        events = _sse_events(await res.aread())

    assert calls == 1
    stats = [data for event, data in events if event == "stats"]
    assert stats
    assert '"truncated": true' in stats[-1]
    assert '"finish_reason": "length"' in stats[-1]
