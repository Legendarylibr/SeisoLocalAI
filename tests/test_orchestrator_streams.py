from __future__ import annotations

import asyncio
import base64

import pytest

from forge.db.crypto import resolve_encryption_key
from forge.db.store import Database
from forge.orchestrators.base import (
    MAX_LOG_LINES,
    MAX_METRIC_POINTS,
    JobStatus,
    Orchestrator,
)

_TEST_KEY = base64.b64encode(b"\x02" * 32).decode()


class DummyOrchestrator(Orchestrator):
    kind = "dummy"

    async def execute(self, job_id: str, payload: dict) -> dict:
        return payload


@pytest.mark.asyncio
async def test_log_stream_removes_subscriber_on_close(tmp_path):
    orchestrator = DummyOrchestrator(tmp_path)
    job_id = orchestrator.create_job(user_id="u1")
    orchestrator._emit_log(job_id, "hello")

    stream = orchestrator.stream_logs(job_id)
    assert len(orchestrator._subscribers[job_id]) == 0

    assert await stream.asend(None) == "hello"
    assert len(orchestrator._subscribers[job_id]) == 1

    await stream.aclose()
    assert len(orchestrator._subscribers[job_id]) == 0


@pytest.mark.asyncio
async def test_metric_stream_removes_subscriber_on_close(tmp_path):
    orchestrator = DummyOrchestrator(tmp_path)
    job_id = orchestrator.create_job(user_id="u1")
    orchestrator._emit_metric(job_id, {"loss": 1.0})

    stream = orchestrator.stream_metrics(job_id)
    assert len(orchestrator._metric_subscribers[job_id]) == 0

    assert await stream.asend(None) == {"loss": 1.0}
    assert len(orchestrator._metric_subscribers[job_id]) == 1

    await stream.aclose()
    assert len(orchestrator._metric_subscribers[job_id]) == 0


@pytest.mark.asyncio
async def test_log_stream_removes_subscriber_after_finished_job(tmp_path):
    orchestrator = DummyOrchestrator(tmp_path)
    job_id = orchestrator.create_job(user_id="u1")
    orchestrator.get_job(job_id).status = JobStatus.COMPLETED  # type: ignore[union-attr]

    assert [line async for line in orchestrator.stream_logs(job_id)] == []
    assert len(orchestrator._subscribers[job_id]) == 0


@pytest.mark.asyncio
async def test_metric_stream_removes_subscriber_after_finished_job(tmp_path):
    orchestrator = DummyOrchestrator(tmp_path)
    job_id = orchestrator.create_job(user_id="u1")
    orchestrator._emit_metric(job_id, {"loss": 1.0})
    orchestrator.get_job(job_id).status = JobStatus.COMPLETED  # type: ignore[union-attr]

    assert [point async for point in orchestrator.stream_metrics(job_id)] == [{"loss": 1.0}]
    assert len(orchestrator._metric_subscribers[job_id]) == 0


def test_log_buffer_is_capped_without_manual_eviction(tmp_path):
    orchestrator = DummyOrchestrator(tmp_path)
    job_id = orchestrator.create_job(user_id="u1")

    for idx in range(MAX_LOG_LINES + 2):
        orchestrator._emit_log(job_id, f"line {idx}")

    assert list(orchestrator._log_buffers[job_id]) == [
        f"line {idx}" for idx in range(2, MAX_LOG_LINES + 2)
    ]


def test_metric_buffer_is_capped_without_manual_eviction(tmp_path):
    orchestrator = DummyOrchestrator(tmp_path)
    job_id = orchestrator.create_job(user_id="u1")

    for idx in range(MAX_METRIC_POINTS + 2):
        orchestrator._emit_metric(job_id, {"step": idx})

    assert list(orchestrator._metric_buffers[job_id]) == [
        {"step": idx} for idx in range(2, MAX_METRIC_POINTS + 2)
    ]


@pytest.mark.asyncio
async def test_live_log_stream_survives_buffer_wrap(tmp_path):
    """Subscribers must keep receiving lines after maxlen deque wraps."""
    orchestrator = DummyOrchestrator(tmp_path)
    job_id = orchestrator.create_job(user_id="u1")

    async def _produce():
        await asyncio.sleep(0.01)
        for idx in range(MAX_LOG_LINES + 5):
            orchestrator._emit_log(job_id, f"line {idx}")
            await asyncio.sleep(0)
        orchestrator.get_job(job_id).status = JobStatus.COMPLETED  # type: ignore[union-attr]
        orchestrator._finish_logs(job_id)

    producer = asyncio.create_task(_produce())
    lines = [line async for line in orchestrator.stream_logs(job_id, replay_buffer=False)]
    await producer
    assert f"line {MAX_LOG_LINES + 4}" in lines
    assert len(lines) >= 5


@pytest.mark.asyncio
async def test_live_only_stream_skips_buffer_replay(tmp_path):
    orchestrator = DummyOrchestrator(tmp_path)
    job_id = orchestrator.create_job(user_id="u1")
    orchestrator._emit_log(job_id, "persisted")
    orchestrator.get_job(job_id).status = JobStatus.COMPLETED  # type: ignore[union-attr]

    assert [line async for line in orchestrator.stream_logs(job_id, replay_buffer=False)] == []


@pytest.mark.asyncio
async def test_durable_job_events_replay_as_sse(tmp_path):
    from forge.api.routes._stream import durable_job_events

    db = Database(
        tmp_path / "forge.db",
        encryption_key=resolve_encryption_key(_TEST_KEY),
        ephemeral=True,
    )
    user = await db.create_user("hashed", "User", email="stream@local.dev")
    await db.append_job_event(
        job_id="job-1",
        user_id=user["id"],
        kind="training",
        event_type="log",
        payload={"line": "hello"},
    )
    await db.append_job_event(
        job_id="job-1",
        user_id=user["id"],
        kind="training",
        event_type="memory_policy",
        payload={"reason": "oom_fallback"},
    )

    events = [event async for event in durable_job_events(db, "job-1", user["id"])]
    assert events[0] == {"event": "log", "data": "hello"}
    assert events[1]["event"] == "memory_policy"
    assert "oom_fallback" in events[1]["data"]
