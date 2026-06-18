"""RL quantization job orchestrator — adaptive_quant research pipeline."""

from __future__ import annotations

from forge.orchestrators._vendor_job import vendor_orchestrator
from seiso.rl_quant.runner import run_rl_quant_job

RLQuantOrchestrator = vendor_orchestrator(
    class_name="RLQuantOrchestrator",
    kind="rl_quant",
    user_id_error="user_id required for RL quant job",
    path_keys=("checkpoint_path", "gguf_path"),
    start_message="Starting adaptive RL quantization pipeline",
    runner=run_rl_quant_job,
    result_log=lambda result: f"Artifacts: {result.get('output_dir')}",
)
