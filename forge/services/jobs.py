"""Job ownership and authorization helpers."""

from __future__ import annotations

from fastapi import HTTPException

from forge.orchestrators.base import Orchestrator


def assert_job_owner(orchestrator: Orchestrator, job_id: str, user_id: str) -> None:
    job = orchestrator.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if not job.user_id or job.user_id != user_id:
        raise HTTPException(403, "Not authorized for this job")
