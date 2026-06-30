from __future__ import annotations

import pytest

from forge.orchestrators.base import JobStatus, Orchestrator


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
    assert orchestrator._subscribers[job_id] == []

    assert await stream.asend(None) == "hello"
    assert len(orchestrator._subscribers[job_id]) == 1

    await stream.aclose()
    assert orchestrator._subscribers[job_id] == []


@pytest.mark.asyncio
async def test_metric_stream_removes_subscriber_on_close(tmp_path):
    orchestrator = DummyOrchestrator(tmp_path)
    job_id = orchestrator.create_job(user_id="u1")
    orchestrator._emit_metric(job_id, {"loss": 1.0})

    stream = orchestrator.stream_metrics(job_id)
    assert orchestrator._metric_subscribers[job_id] == []

    assert await stream.asend(None) == {"loss": 1.0}
    assert len(orchestrator._metric_subscribers[job_id]) == 1

    await stream.aclose()
    assert orchestrator._metric_subscribers[job_id] == []


@pytest.mark.asyncio
async def test_log_stream_removes_subscriber_after_finished_job(tmp_path):
    orchestrator = DummyOrchestrator(tmp_path)
    job_id = orchestrator.create_job(user_id="u1")
    orchestrator.get_job(job_id).status = JobStatus.COMPLETED  # type: ignore[union-attr]

    assert [line async for line in orchestrator.stream_logs(job_id)] == []
    assert orchestrator._subscribers[job_id] == []
