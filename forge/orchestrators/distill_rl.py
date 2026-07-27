"""Distill-RL job orchestrator — teacher distillation + preference DPO pipeline."""

from __future__ import annotations

from forge.orchestrators._bundled_job import BundledJobContract, bundled_orchestrator
from seiso.distill_rl.runner import run_distill_rl_job

DistillRLOrchestrator = bundled_orchestrator(
    class_name="DistillRLOrchestrator",
    kind="distill_rl",
    user_id_error="user_id required for distill-RL job",
    path_keys=("distilled_path", "prompt_library"),
    local_path_keys=("dataset_ref", "hf_dataset"),
    start_message="Starting distill-RL pipeline (distill → rollout → DPO → evaluate)",
    runner=run_distill_rl_job,
    result_log=lambda result: f"Artifacts: {result.get('output_dir')}",
    contract=BundledJobContract(requires_manifest=True),
)
