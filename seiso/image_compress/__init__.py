"""Stable Diffusion image compression — distillation, pruning, quantization pipeline."""

from seiso.image_compress.bootstrap import (
    ensure_sd_compress_importable,
    require_sd_compress,
    vendor_root,
)
from seiso.image_compress.config_builder import PRESETS, STAGE_ORDER, build_pipeline_config
from seiso.image_compress.runner import run_image_compress_job

__all__ = [
    "PRESETS",
    "STAGE_ORDER",
    "build_pipeline_config",
    "ensure_sd_compress_importable",
    "require_sd_compress",
    "run_image_compress_job",
    "vendor_root",
]
