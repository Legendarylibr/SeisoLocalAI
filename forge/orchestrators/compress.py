"""Code Llama compression job orchestrator."""

from __future__ import annotations

from typing import Any

from forge.orchestrators._vendor_job import run_vendor_job
from forge.orchestrators.base import Orchestrator
from seiso.compress.runner import run_compress_job


class CompressOrchestrator(Orchestrator):
    kind = "compress"

    async def execute(self, job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return await run_vendor_job(
            self,
            job_id,
            payload,
            user_id_error="user_id required for compression job",
            path_keys=("model_dir",),
            start_message="Starting Code Llama compression pipeline",
            runner=run_compress_job,
            result_log=lambda result: f"Run directory: {result.get('run_dir')}",
        )
