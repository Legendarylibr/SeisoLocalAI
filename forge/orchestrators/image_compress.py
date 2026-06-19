"""Stable Diffusion image compression job orchestrator."""

from __future__ import annotations

from forge.orchestrators._vendor_job import vendor_orchestrator
from seiso.image_compress.runner import run_image_compress_job

ImageCompressOrchestrator = vendor_orchestrator(
    class_name="ImageCompressOrchestrator",
    kind="image_compress",
    user_id_error="user_id required for image compression job",
    path_keys=("model_dir", "data_path"),
    start_message="Starting Stable Diffusion image compression pipeline",
    runner=run_image_compress_job,
    result_log=lambda result: f"Run directory: {result.get('run_dir')}",
)
