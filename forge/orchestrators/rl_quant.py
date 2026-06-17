"""RL quantization job orchestrator — adaptive_quant research pipeline."""

from __future__ import annotations

from typing import Any

from forge.orchestrators._vendor_job import run_vendor_job
from forge.orchestrators.base import Orchestrator
from seiso.rl_quant.runner import run_rl_quant_job


class RLQuantOrchestrator(Orchestrator):
    kind = "rl_quant"

    async def execute(self, job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return await run_vendor_job(
            self,
            job_id,
            payload,
            user_id_error="user_id required for RL quant job",
            path_keys=("checkpoint_path", "gguf_path"),
            start_message="Starting adaptive RL quantization pipeline",
            runner=run_rl_quant_job,
            result_log=lambda result: f"Artifacts: {result.get('output_dir')}",
        )
