"""Local-only hardware detection and live metrics — Forge extensions on seiso.hardware."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from seiso.hardware import (
    FIT_RANK,
    TIER_LABELS,
    GuideStep,
    HardwareTier,
    assess_catalog_fit,
    assess_hardware_fit,
    assess_inference_option_fit,
    build_guidance,
    classify_tier,
    detect_gpus,
    effective_budget_mb,
    live_metrics,
    memory_headroom_label,
    preferred_inference_backend,
    training_defaults,
    vram_headroom_mb,
)
from seiso.hardware import (
    hardware_profile as _core_hardware_profile,
)
from seiso.hardware.fit import format_catalog_note
from seiso.hardware.profile import enrich_profile_base
from seiso.inference.backends import BACKEND_LABELS
from seiso.memory.estimates import estimate_chat_vram_gb, estimate_gguf_download_bytes

_RECOMMENDED_REPO_TTL_SEC = 300.0
_recommended_repo_cache: dict[tuple, tuple[float, str | None]] = {}


def enrich_profile(profile: dict[str, Any]) -> dict[str, Any]:
    """Add tier, training defaults, catalog recommendations, and backend labels."""
    enriched = enrich_profile_base(profile)
    recommended_chat = recommended_catalog_repo(enriched, task="chat")
    return {
        **enriched,
        "recommended_chat_repo": recommended_chat,
        "recommended_train_repo": recommended_chat,
        "inference_backend_labels": dict(BACKEND_LABELS),
    }


def hardware_profile(*, force_refresh: bool = False) -> dict[str, Any]:
    """Full hardware profile for Forge API — includes Hub recommendations."""
    return enrich_profile(_core_hardware_profile(force_refresh=force_refresh))


def enrich_catalog_models(
    models: list[dict[str, Any]],
    profile: dict[str, Any],
    *,
    token: str | None = None,
    fetch_sizes: bool = True,
    diversify: bool = False,
) -> list[dict[str, Any]]:
    from forge.services.hf_hub import resolve_gguf_artifact
    from seiso.models.catalog import diversify_by_family, get_by_repo

    download_info: dict[str, dict[str, Any]] = {}
    download_errors: dict[str, str] = {}
    if models and fetch_sizes:
        candidates = models if token else models[:16]
        workers = min(3 if token else 2, len(candidates))

        def fetch_info(repo_id: str) -> tuple[str, dict[str, Any] | None, str | None]:
            try:
                return (
                    repo_id,
                    resolve_gguf_artifact(
                        repo_id,
                        entry=get_by_repo(repo_id),
                        token=token,
                    ),
                    None,
                )
            except Exception as exc:
                return repo_id, None, str(exc)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(fetch_info, m["repo_id"]): m["repo_id"] for m in candidates}
            for future in as_completed(futures):
                repo_id, info, error = future.result()
                if info:
                    download_info[repo_id] = info
                elif error:
                    download_errors[repo_id] = error

    headroom_gb = round(vram_headroom_mb(profile) / 1024, 1)
    tier = classify_tier(profile)
    enriched: list[dict[str, Any]] = []
    for m in models:
        fit = assess_catalog_fit(m, profile)
        row = {**m, **fit}
        info = download_info.get(m["repo_id"])
        if info and info.get("size_bytes"):
            download_bytes = int(info["size_bytes"])
            actual_fit = assess_hardware_fit(
                round(download_bytes / (1024**3) + 0.8, 2),
                profile,
                mode="chat",
            )
            row.update(actual_fit)
            row["download_bytes"] = download_bytes
            row["download_bytes_estimated"] = False
            row["gguf_repo"] = info["gguf_repo"]
            row["gguf_file"] = info["filename"]
            row["download_mirror_verified"] = True
            row["download_available"] = True
        elif m["repo_id"] in download_errors:
            download_bytes = estimate_gguf_download_bytes(
                m["params"],
                quant=m.get("quant", "Q4_K_M"),
                tags=m.get("tags", ()),
                repo_id=m.get("repo_id", ""),
            )
            row["download_bytes"] = download_bytes
            row["download_bytes_estimated"] = True
            row["download_mirror_verified"] = False
            row["download_error"] = download_errors[m["repo_id"]]
            row["download_available"] = (
                m.get("task") != "embedding" and "trusted GGUF" not in download_errors[m["repo_id"]]
            )
        elif m.get("task") != "embedding":
            download_bytes = estimate_gguf_download_bytes(
                m["params"],
                quant=m.get("quant", "Q4_K_M"),
                tags=m.get("tags", ()),
                repo_id=m.get("repo_id", ""),
            )
            row["download_bytes"] = download_bytes
            row["download_bytes_estimated"] = True
            row["download_mirror_verified"] = False
            row["download_available"] = True
        else:
            download_bytes = 0
            row["download_available"] = False

        if download_bytes > 0:
            row["hardware_note"] = format_catalog_note(
                est_vram_gb=fit["est_vram_mb"] / 1024,
                download_bytes=download_bytes,
                headroom_gb=headroom_gb,
                fit=fit["hardware_fit"],
                tier=tier,
            )
        enriched.append(row)

    enriched.sort(
        key=lambda m: (
            -(m.get("priority") or 0),
            -m.get("hardware_fit_rank", 0),
            m.get("name", ""),
        )
    )
    if diversify:
        enriched = diversify_by_family(enriched)
        indexed = list(enumerate(enriched))
        indexed.sort(
            key=lambda m: (
                -m[1].get("hardware_fit_rank", 0),
                m[0],
            )
        )
        enriched = [m for _, m in indexed]
    return enriched


def recommended_catalog_repo(profile: dict[str, Any], *, task: str = "chat") -> str | None:
    from seiso.models.catalog import HubSearchError, search_catalog

    tier = classify_tier(profile)
    budget = effective_budget_mb(profile)
    cache_key = (tier.value, budget, task)
    now = time.monotonic()
    cached = _recommended_repo_cache.get(cache_key)
    if cached and now - cached[0] < _RECOMMENDED_REPO_TTL_SEC:
        return cached[1]

    try:
        models = search_catalog(task=task).models if task else search_catalog().models
    except HubSearchError:
        _recommended_repo_cache[cache_key] = (now, None)
        return None

    models = enrich_catalog_models(models, profile, fetch_sizes=False, diversify=True)
    result: str | None = None
    for m in models:
        if m.get("hardware_fit") in ("ideal", "good") and m.get("task") != "embedding":
            result = m["repo_id"]
            break
    if result is None:
        for m in models:
            if m.get("task") != "embedding":
                result = m["repo_id"]
                break

    _recommended_repo_cache[cache_key] = (now, result)
    return result


def largest_fitting_catalog_repo(profile: dict[str, Any], *, task: str = "chat") -> str | None:
    """Largest catalog model with ideal/good hardware fit."""
    return recommended_catalog_repo(profile, task=task)


def build_vram_status(orchestrator: Any) -> dict[str, Any]:
    """Unified VRAM/RAM status for API responses."""
    from seiso.hardware.tiers import HardwareTier, classify_tier, vram_headroom_mb
    from seiso.memory.platform_profile import memory_profile_label

    profile = hardware_profile(force_refresh=False)
    tier = classify_tier(profile)
    headroom = vram_headroom_mb(profile)
    local = orchestrator._runner._pool.status()
    return {
        "local": local,
        "ollama_model": orchestrator.active_ollama_model,
        "headroom_mb": headroom,
        "memory_label": memory_headroom_label(profile),
        "ram_gb": profile.get("ram_gb"),
        "apple_unified": tier == HardwareTier.APPLE_UNIFIED,
        "tier": tier.value,
        "memory_profile": memory_profile_label(profile),
        "recommended_max_chat": largest_fitting_catalog_repo(profile, task="chat"),
        "active_model": local.get("active_model") or orchestrator.active_ollama_model,
    }


def hardware_summary(profile: dict[str, Any]) -> dict[str, Any]:
    """Compact summary safe to embed in API responses."""
    tier = classify_tier(profile)
    preferred = preferred_inference_backend(profile)
    return {
        "tier": tier.value,
        "tier_label": TIER_LABELS[tier],
        "backend": profile.get("backend"),
        "ram_gb": profile.get("ram_gb"),
        "gpu_count": len(profile.get("gpus") or []),
        "effective_vram_mb": effective_budget_mb(profile),
        "vram_headroom_mb": vram_headroom_mb(profile),
        "memory_headroom_label": memory_headroom_label(profile),
        "preferred_inference_backend": preferred,
        "preferred_inference_backend_label": BACKEND_LABELS.get(preferred, preferred),
        "inference_backend_labels": dict(BACKEND_LABELS),
        "local_only": True,
    }


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
    "enrich_catalog_models",
    "enrich_profile",
    "estimate_chat_vram_gb",
    "estimate_gguf_download_bytes",
    "format_catalog_note",
    "hardware_profile",
    "hardware_summary",
    "largest_fitting_catalog_repo",
    "build_vram_status",
    "live_metrics",
    "memory_headroom_label",
    "preferred_inference_backend",
    "recommended_catalog_repo",
    "training_defaults",
    "vram_headroom_mb",
]
