from __future__ import annotations

import pytest

from forge.orchestrators.base import (
    MAX_LOG_LINES,
    MAX_METRIC_POINTS,
    JobStatus,
    Orchestrator,
)


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

    assert [point async for point in orchestrator.stream_metrics(job_id)] == [
        {"loss": 1.0}
    ]
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
