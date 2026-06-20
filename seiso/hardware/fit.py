"""Hardware fit assessment for models and inference options."""

from __future__ import annotations

from typing import Any

from seiso.hardware.tiers import FIT_RANK, HardwareTier, classify_tier, vram_headroom_mb
from seiso.memory.estimates import estimate_chat_vram_gb, guess_params_from_name


def assess_hardware_fit(
    est_vram_gb: float,
    profile: dict[str, Any],
    *,
    mode: str = "chat",
) -> dict[str, Any]:
    """Return fit label + short note — never leaves the machine."""
    headroom_mb = vram_headroom_mb(profile)
    est_mb = int(est_vram_gb * 1024)
    tier = classify_tier(profile)

    if mode == "train":
        est_mb = int(est_mb * 2.2)

    ratio = est_mb / headroom_mb if headroom_mb > 0 else 99.0

    if ratio <= 0.65:
        fit, label = "ideal", "Ideal fit"
    elif ratio <= 0.95:
        fit, label = "good", "Good fit"
    elif ratio <= 1.15:
        fit, label = "tight", "Tight fit"
    else:
        fit, label = "unlikely", "May not fit"

    if tier == HardwareTier.CPU_ONLY and est_mb > 4096:
        fit, label = "unlikely", "CPU — try ≤3B Q4"

    if tier == HardwareTier.APPLE_UNIFIED and headroom_mb < 12288 and est_mb > 5120:
        fit, label = "tight", "Tight — use Q4 GGUF + llama.cpp"
    if tier == HardwareTier.APPLE_UNIFIED and headroom_mb < 8192 and est_mb > 4096:
        fit, label = "unlikely", "Low memory — try ≤3B Q4"

    headroom_gb = round(headroom_mb / 1024, 1)
    note = f"~{est_vram_gb:.1f} GB est. · {headroom_gb} GB free on this machine"
    if fit == "unlikely" and tier != HardwareTier.CPU_ONLY:
        note = f"Needs ~{est_vram_gb:.1f} GB — you have ~{headroom_gb} GB free"

    blocked = headroom_mb > 0 and est_mb > int(headroom_mb * 1.12)
    block_reason = None
    if blocked:
        block_reason = (
            f"Needs ~{est_vram_gb:.1f} GB at runtime but only ~{headroom_gb} GB is free on this machine. "
            "Choose a smaller or more quantized model."
        )

    return {
        "hardware_fit": fit,
        "hardware_fit_label": label,
        "est_vram_mb": est_mb,
        "hardware_note": note,
        "hardware_fit_rank": FIT_RANK[fit],
        "memory_load_blocked": blocked,
        "memory_load_blocked_reason": block_reason,
    }


def assess_catalog_fit(model: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    est_gb = estimate_chat_vram_gb(
        model["params"],
        quant=model.get("quant", "Q4_K_M"),
        tags=model.get("tags", ()),
        repo_id=model.get("repo_id", ""),
    )
    mode = "train" if model.get("task") in ("base",) else "chat"
    return assess_hardware_fit(est_gb, profile, mode=mode)


def assess_inference_option_fit(option: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    size_bytes = int(option.get("size_bytes") or 0)
    name = option.get("name") or ""
    if size_bytes > 0:
        est_gb = round(size_bytes / (1024**3) + 0.8, 2)
    else:
        guessed = guess_params_from_name(name)
        if guessed:
            est_gb = estimate_chat_vram_gb(f"{guessed}B")
        elif option.get("kind") == "ollama":
            guessed = guess_params_from_name(name.split(":")[0])
            est_gb = estimate_chat_vram_gb(f"{guessed or 7}B") if guessed else 5.0
        else:
            est_gb = 6.0
    return assess_hardware_fit(est_gb, profile)


def format_catalog_note(
    *,
    est_vram_gb: float,
    download_bytes: int,
    headroom_gb: float,
    fit: str,
    tier: HardwareTier,
) -> str:
    dl = f"Download ~{download_bytes / (1024**3):.1f} GB · " if download_bytes > 0 else ""
    runtime = f"Runtime ~{est_vram_gb:.1f} GB est. · "
    if fit == "unlikely" and tier != HardwareTier.CPU_ONLY:
        return f"{dl}{runtime}Needs ~{est_vram_gb:.1f} GB at runtime — you have ~{headroom_gb} GB free"
    return f"{dl}{runtime}{headroom_gb} GB free on this machine"
