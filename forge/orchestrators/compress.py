"""LLM compression job orchestrator."""

from __future__ import annotations

from forge.orchestrators._bundled_job import BundledJobContract, bundled_orchestrator
from seiso.compress.runner import run_compress_job

CompressOrchestrator = bundled_orchestrator(
    class_name="CompressOrchestrator",
    kind="compress",
    user_id_error="user_id required for compression job",
    path_keys=("model_dir",),
    start_message="Starting LLM compression pipeline",
    runner=run_compress_job,
    result_log=lambda result: f"Run directory: {result.get('run_dir')}",
    contract=BundledJobContract(requires_manifest=True),
)
