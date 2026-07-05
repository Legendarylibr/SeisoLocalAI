"""Hardware fit assessment for models and inference options."""

from __future__ import annotations

from typing import Any

from seiso.hardware.tiers import (
    FIT_RANK,
    HardwareTier,
    classify_tier,
    fit_headroom_mb,
    ram_headroom_mb,
    vram_headroom_mb,
)
from seiso.platform import is_native_linux_nvidia
from seiso.memory.estimates import estimate_chat_vram_gb, guess_params_from_name

_LOAD_RESERVE_RATIO = 0.02
_LOAD_MIN_RESERVE_MB = 256


def _usable_load_budget_mb(*, capacity_mb: int, free_mb: int) -> int:
    """Full measured headroom minus a small reserve for allocator/runtime overhead."""
    raw_budget = free_mb if free_mb > 0 else capacity_mb
    if raw_budget <= 0:
        return 0
    reserve = max(_LOAD_MIN_RESERVE_MB, int(raw_budget * _LOAD_RESERVE_RATIO))
    return max(0, raw_budget - reserve)


def assess_hardware_fit(
    est_vram_gb: float,
    profile: dict[str, Any],
    *,
    mode: str = "chat",
) -> dict[str, Any]:
    """Return fit label + short note — never leaves the machine."""
    capacity_mb = fit_headroom_mb(profile)
    free_mb = vram_headroom_mb(profile)
    est_mb = int(est_vram_gb * 1024)
    tier = classify_tier(profile)

    # Note: train-mode overhead is already applied by estimate_path_vram_mb()
    # via _TRAINING_OVERHEAD_RATIO.  Do not multiply again here.

    ratio = est_mb / capacity_mb if capacity_mb > 0 else 99.0

    if ratio <= 0.65:
        fit, label = "ideal", "Ideal fit"
    elif ratio <= 0.95:
        fit, label = "good", "Good fit"
    elif ratio <= 1.05:
        fit, label = "tight", "Tight fit"
    else:
        fit, label = "unlikely", "May not fit"

    if tier == HardwareTier.APPLE_UNIFIED and capacity_mb < 12288 and est_mb > 5120:
        fit, label = "tight", "Tight — use Q4 GGUF + llama.cpp"
    if tier == HardwareTier.APPLE_UNIFIED and capacity_mb < 8192 and est_mb > 4096:
        fit, label = "unlikely", "Low memory — try ≤3B Q4"

    # Warn when the model fits the GPU but current free VRAM is low (other apps/models).
    if (
        fit in {"ideal", "good", "tight"}
        and free_mb > 0
        and est_mb > int(free_mb * 1.05)
        and capacity_mb > free_mb
    ):
        fit, label = "tight", "Tight fit — free VRAM is low; close other GPU apps first"

    ram_free_mb = ram_headroom_mb(profile)
    ram_budget_mb = _usable_load_budget_mb(
        capacity_mb=ram_free_mb, free_mb=ram_free_mb
    )
    vram_budget_mb = _usable_load_budget_mb(
        capacity_mb=capacity_mb, free_mb=free_mb
    )
    native_linux_gguf_chat = (
        mode == "chat"
        and is_native_linux_nvidia(profile=profile)
        and tier not in (HardwareTier.APPLE_UNIFIED, HardwareTier.CPU_ONLY)
    )
    if native_linux_gguf_chat:
        # GGUF mmap on Linux needs host RAM; full GPU offload is tried at load time.
        load_budget_mb = ram_budget_mb
        memory_label = "RAM"
    else:
        load_budget_mb = vram_budget_mb
        memory_label = (
            "RAM"
            if tier in (HardwareTier.APPLE_UNIFIED, HardwareTier.CPU_ONLY)
            else "VRAM"
        )

    capacity_gb = round(capacity_mb / 1024, 1)
    free_gb = round(free_mb / 1024, 1)
    ram_free_gb = round(ram_free_mb / 1024, 1)
    load_budget_gb = round(load_budget_mb / 1024, 1)
    note = f"~{est_vram_gb:.1f} GB est. · {free_gb} GB free now · {capacity_gb} GB GPU budget"
    if native_linux_gguf_chat:
        note = (
            f"~{est_vram_gb:.1f} GB est. · {ram_free_gb} GB RAM free · "
            f"{free_gb} GB VRAM free"
        )
    elif fit == "unlikely" and tier != HardwareTier.CPU_ONLY:
        note = f"Needs ~{est_vram_gb:.1f} GB — GPU budget ~{capacity_gb} GB"

    blocked = load_budget_mb > 0 and est_mb > load_budget_mb
    block_reason = None
    if blocked:
        label = "Blocked — would exceed available memory"
        block_reason = (
            f"Needs ~{est_vram_gb:.1f} GB at runtime but only ~{load_budget_gb} GB "
            f"{memory_label} is safely available right now. Free memory or choose a "
            "smaller/more quantized model."
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


def assess_catalog_fit(
    model: dict[str, Any], profile: dict[str, Any]
) -> dict[str, Any]:
    tags = tuple(model.get("tags") or ())
    download_bytes = int(model.get("download_bytes") or 0)
    if download_bytes > 0:
        est_gb = round(download_bytes / (1024**3) + 0.8, 2)
    elif "moe" in tags:
        from seiso.memory.estimates import estimate_gguf_download_bytes

        est_gb = round(
            estimate_gguf_download_bytes(
                model["params"],
                quant=model.get("quant", "Q4_K_M"),
                tags=tags,
                repo_id=model.get("repo_id", ""),
            )
            / (1024**3)
            + 0.8,
            2,
        )
    else:
        est_gb = estimate_chat_vram_gb(
            model["params"],
            quant=model.get("quant", "Q4_K_M"),
            tags=tags,
            repo_id=model.get("repo_id", ""),
        )
    mode = "train" if model.get("task") in ("base",) else "chat"
    result = assess_hardware_fit(est_gb, profile, mode=mode)
    if "moe" in tags and download_bytes <= 0:
        note = result.get("hardware_note") or ""
        moe_hint = "MoE — load needs full GGUF in RAM (mmap); active experts are smaller at runtime."
        result["hardware_note"] = f"{note} · {moe_hint}" if note else moe_hint
    return result


def assess_inference_option_fit(
    option: dict[str, Any], profile: dict[str, Any]
) -> dict[str, Any]:
    size_bytes = int(option.get("size_bytes") or 0)
    name = option.get("name") or ""
    if size_bytes > 0:
        est_gb = round(size_bytes / (1024**3) + 0.8, 2)
    else:
        guessed = guess_params_from_name(name)
        est_gb = estimate_chat_vram_gb(f"{guessed}B") if guessed else 6.0
    return assess_hardware_fit(est_gb, profile)


def format_catalog_note(
    *,
    est_vram_gb: float,
    download_bytes: int,
    headroom_gb: float,
    fit: str,
    tier: HardwareTier,
) -> str:
    dl = (
        f"Download ~{download_bytes / (1024**3):.1f} GB · "
        if download_bytes > 0
        else ""
    )
    runtime = f"Runtime ~{est_vram_gb:.1f} GB est. · "
    if fit == "unlikely" and tier != HardwareTier.CPU_ONLY:
        return f"{dl}{runtime}Needs ~{est_vram_gb:.1f} GB at runtime — you have ~{headroom_gb} GB free"
    return f"{dl}{runtime}{headroom_gb} GB free on this machine"
