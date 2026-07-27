"""Hardware-aware training and inference backend defaults."""

from __future__ import annotations

from typing import Any

from seiso.hardware.tiers import (
    TIER_LABELS,
    HardwareTier,
    classify_tier,
    vram_headroom_mb,
)
from seiso.inference.backends import InferenceBackend
from seiso.models.loader import Backend


def preferred_inference_backend(profile: dict[str, Any]) -> str:
    tier = classify_tier(profile)
    try:
        backend = Backend(profile.get("backend", "cpu"))
    except ValueError:
        backend = Backend.CPU

    if _profile_has_nvidia(profile):
        return str(InferenceBackend.LLAMASWAP)
    if _profile_has_rocm(profile):
        return str(InferenceBackend.TORCH)

    if tier == HardwareTier.CPU_ONLY:
        return str(InferenceBackend.LLAMACPP)
    if tier == HardwareTier.EDGE:
        return str(InferenceBackend.LLAMACPP)
    if tier == HardwareTier.APPLE_UNIFIED:
        # Tight unified RAM: prefer GGUF/llama.cpp (SKIP_MLX_PROBE seeds this).
        ram_gb = float(profile.get("ram_gb") or 0)
        if ram_gb > 0 and ram_gb <= 24:
            return str(InferenceBackend.LLAMACPP)
        if backend == Backend.MLX:
            return str(InferenceBackend.MLX)
        return str(InferenceBackend.LLAMACPP)
    if backend == Backend.MLX:
        return str(InferenceBackend.MLX)
    return str(InferenceBackend.LLAMACPP)


def _profile_has_nvidia(profile: dict[str, Any]) -> bool:
    if str(profile.get("backend", "")).lower() in {"cuda", "torch"}:
        return True
    for gpu in profile.get("gpus") or []:
        text = " ".join(
            str(gpu.get(key, "")) for key in ("name", "vendor", "type", "backend")
        ).lower()
        if "nvidia" in text or "cuda" in text:
            return True
    return False


def _profile_has_rocm(profile: dict[str, Any]) -> bool:
    if str(profile.get("backend", "")).lower() in {"rocm", "hip"}:
        return True
    for gpu in profile.get("gpus") or []:
        text = " ".join(
            str(gpu.get(key, "")) for key in ("name", "vendor", "type", "backend")
        ).lower()
        if any(marker in text for marker in ("amd", "rocm", "radeon", "hip")):
            return True
    return False


def training_defaults(profile: dict[str, Any]) -> dict[str, Any]:
    from seiso.training.platform_caps import training_capabilities

    tier = classify_tier(profile)
    headroom = vram_headroom_mb(profile)
    ram = float(profile.get("ram_gb") or 0)
    caps = training_capabilities()

    if tier in (HardwareTier.WORKSTATION, HardwareTier.CAPABLE) and headroom >= 16000:
        batch, accum, max_seq, max_params = 2, 4, 4096, "14B"
    elif headroom >= 10000 or tier == HardwareTier.APPLE_UNIFIED:
        batch, accum, max_seq, max_params = 1, 8, 2048, "7B"
    elif headroom >= 6000:
        batch, accum, max_seq, max_params = 1, 16, 2048, "3B"
    else:
        batch, accum, max_seq, max_params = 1, 16, 1024, "1B"

    quant = caps["recommended_quant"]
    note = f"Tuned for {TIER_LABELS[tier]} ({ram:.0f} GB RAM, ~{headroom // 1024} GB free)"
    if not caps["supports_qlora"]:
        note += " — use 16-bit LoRA on macOS (no bitsandbytes)"
    if caps["fused_kernels_available"]:
        note += f" — fused kernels via {caps['kernel_backend']}"
    low_vram = headroom > 0 and headroom < 8192
    if low_vram:
        note += " — lean / low-VRAM mode recommended (<8 GB free)"
    attn = str(caps.get("attn_implementation") or "sdpa")
    if caps.get("flash_attn_available"):
        note += f" — attention {attn}"
    elif caps.get("sdpa_available"):
        note += " — attention SDPA (install flash-attn for longer context)"

    packing = bool(caps.get("recommend_sequence_packing"))
    padding_free = packing and bool(caps.get("flash_attn_available"))

    return {
        "batch_size": batch,
        "gradient_accumulation_steps": accum,
        "max_seq_length": max_seq,
        "quant": quant,
        "method": "lora",
        "gradient_checkpointing": True,
        "max_recommended_params": max_params,
        "use_fused_kernels": caps["fused_kernels_available"],
        "use_fused_ce": caps["fused_ce_available"],
        "kernel_low_vram": low_vram,
        "kernel_backend": caps["kernel_backend"],
        "attn_implementation": attn,
        "flash_attn_available": bool(caps.get("flash_attn_available")),
        "packing": packing,
        "padding_free": padding_free,
        "train_platform": caps["train_platform"],
        "multi_gpu_available": caps["multi_gpu_available"],
        "note": note,
    }
