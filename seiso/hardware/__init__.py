"""Local hardware detection, fit heuristics, and training defaults — no Forge dependency."""

from seiso.hardware.fit import (
    assess_catalog_fit,
    assess_hardware_fit,
    assess_inference_option_fit,
)
from seiso.hardware.guidance import GuideStep, build_guidance
from seiso.hardware.profile import detect_gpus, hardware_profile, live_metrics
from seiso.hardware.tiers import (
    FIT_RANK,
    TIER_LABELS,
    HardwareTier,
    classify_tier,
    effective_budget_mb,
    memory_headroom_label,
    vram_headroom_mb,
)
from seiso.hardware.training import preferred_inference_backend, training_defaults

__all__ = [
    "FIT_RANK",
    "TIER_LABELS",
    "GuideStep",
    "HardwareTier",
    "assess_catalog_fit",
    "assess_hardware_fit",
    "assess_inference_option_fit",
    "build_guidance",
    "classify_tier",
    "detect_gpus",
    "effective_budget_mb",
    "hardware_profile",
    "live_metrics",
    "memory_headroom_label",
    "preferred_inference_backend",
    "training_defaults",
    "vram_headroom_mb",
]
