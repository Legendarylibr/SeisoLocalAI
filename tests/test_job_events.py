"""Job event / orchestrator durable-log regressions."""

from __future__ import annotations

from pathlib import Path

import pytest

@pytest.mark.asyncio
async def test_job_log_event_gen_skips_result_when_cancelled(tmp_path: Path):
    from forge.api.routes._stream import job_log_event_gen
    from forge.orchestrators.base import JobRecord, JobStatus, Orchestrator

    class _Orch(Orchestrator):
        kind = "test"

        async def execute(self, job_id: str, payload: dict) -> dict:
            return {}

    orch = _Orch(tmp_path)
    job_id = "job-cancel-result"
    rec = JobRecord(id=job_id, kind="test", user_id="u1")
    rec.status = JobStatus.CANCELLED
    rec.result = {"model_dir": "/tmp/x"}
    orch._jobs[job_id] = rec

    events = [event async for event in job_log_event_gen(orch, job_id)]
    assert not any(e.get("event") == "result" for e in events)

@pytest.mark.asyncio
async def test_durable_job_events_skip_result_after_cancel(tmp_path: Path):
    from forge.api.routes._stream import durable_job_events
    from forge.db.crypto import generate_encryption_key
    from forge.db.store import Database

    db = Database(
        tmp_path / "forge.db",
        encryption_key=generate_encryption_key(),
        ephemeral=True,
    )
    try:
        user = await db.create_user("hashed", "User", email="cancel-stream@local.dev")
        uid = user["id"]
        await db.append_job_event(
            job_id="j1",
            user_id=uid,
            kind="train",
            event_type="result",
            payload={"checkpoint_path": "/tmp/x"},
        )
        await db.append_job_event(
            job_id="j1",
            user_id=uid,
            kind="train",
            event_type="status",
            payload={"status": "cancelled"},
        )
        events = [e async for e in durable_job_events(db, "j1", uid)]
        assert any(e.get("event") == "status" for e in events)
        assert not any(e.get("event") == "result" for e in events)
    finally:
        await db.close()

def test_bundled_result_rejects_failed_manifest(tmp_path: Path):
    from forge.orchestrators._bundled_job import (
        BundledJobContract,
        validate_bundled_result,
    )

    user_id = "user-1"
    run_dir = tmp_path / "compress" / user_id / "runs" / "run-a"
    run_dir.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="Manifest verification failed"):
        validate_bundled_result(
            tmp_path,
            user_id,
            {
                "run_dir": str(run_dir),
                "manifest": {"ok": False, "error": "hash mismatch"},
            },
            BundledJobContract(requires_manifest=True),
        )

