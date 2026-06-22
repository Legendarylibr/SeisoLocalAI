"""Unified inference model list — HF Hub inventory and CLI paths."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from forge.db.store import Database
from forge.services.artifact_integrity import inventory_gguf_is_complete
from forge.services.hardware import (
    HardwareTier,
    assess_inference_option_fit,
    classify_tier,
    hardware_profile,
    preferred_inference_backend,
    vram_headroom_mb,
)
from forge.services.hf_connectivity import check_inference_runtime
from forge.services.hf_hub import get_gguf_file_size_bytes
from seiso.inference.backends import (
    BACKEND_LABELS,
    BACKEND_LLAMACPP,
    BACKEND_MLX,
    BACKEND_TORCH,
    available_backends,
    recommend_backend,
)

logger = logging.getLogger(__name__)

_OPTIONS_CACHE_TTL_S = 5.0
_options_cache: dict[tuple, tuple[float, list[dict[str, Any]]]] = {}

SOURCE_LABELS = {
    "hf:": "Hugging Face Hub",
    "scan": "CLI scan",
    "manual": "CLI path",
    "training": "Fine-tune output",
    "export": "Export output",
}


def _pick_default_backend(
    backends: list[str],
    profile: dict[str, Any] | None,
) -> str:
    """Hardware-aware default when a model supports multiple inference engines."""
    if not backends:
        return BACKEND_LLAMACPP
    if len(backends) == 1:
        return backends[0]
    if profile:
        preferred = preferred_inference_backend(profile)
        if preferred in backends:
            return preferred
        if BACKEND_LLAMACPP in backends:
            tier = classify_tier(profile)
            if tier in (HardwareTier.CPU_ONLY, HardwareTier.EDGE):
                return BACKEND_LLAMACPP
            if vram_headroom_mb(profile) >= 8000:
                return backends[0]
            return BACKEND_LLAMACPP
    return backends[0]


def _source_label(source: str | None) -> str:
    raw = source or "manual"
    if raw.startswith("hf:"):
        return SOURCE_LABELS["hf:"]
    return SOURCE_LABELS.get(raw, raw)


def _installed_backends() -> dict[str, bool]:
    runtime = check_inference_runtime()
    return {
        BACKEND_LLAMACPP: runtime.llamacpp,
        BACKEND_MLX: runtime.mlx,
        BACKEND_TORCH: runtime.torch,
    }


def _filter_installed_backends(backends: list[str], installed: dict[str, bool]) -> list[str]:
    return [b for b in backends if installed.get(b, False)]


def _inventory_artifact_is_complete(row: dict[str, Any], metadata: dict[str, Any]) -> bool:
    return inventory_gguf_is_complete(
        row,
        metadata,
        size_lookup=get_gguf_file_size_bytes,
    )


def _build_local_option(
    row: dict[str, Any],
    *,
    installed: dict[str, bool],
    profile: dict[str, Any] | None,
) -> dict[str, Any] | None:
    metadata = json.loads(row.get("metadata_json") or "{}")
    if not _inventory_artifact_is_complete(row, metadata):
        return None
    backends = _filter_installed_backends(
        available_backends(
            model_path=row["path"],
            model_format=row.get("format"),
        ),
        installed,
    )
    opt: dict[str, Any] = {
        "id": row["id"],
        "kind": "local",
        "name": row["name"],
        "source": row.get("source") or "manual",
        "source_label": _source_label(row.get("source")),
        "format": row.get("format"),
        "path": row["path"],
        "default_backend": _pick_default_backend(backends, profile) if backends else "",
        "backends": backends,
        "backend_labels": {b: BACKEND_LABELS.get(b, b) for b in backends},
        "size_bytes": row.get("size_bytes", 0),
        "metadata": metadata,
    }
    from forge.services.inference_variants import extract_quant_label, variant_group_key

    opt["quant_label"] = extract_quant_label(
        name=row["name"],
        path=row["path"],
        metadata=metadata,
    )
    opt["variant_group"] = variant_group_key(opt)
    if not backends and (row.get("format") or "").lower() == "gguf":
        runtime = check_inference_runtime()
        opt["install_hints"] = [
            hint for hint in runtime.install_hints if "llama" in hint.lower()
        ] or ['pip install -e ".[llamacpp]"  # GGUF chat via llama.cpp']
    if profile:
        opt.update(assess_inference_option_fit(opt, profile))
    return opt


def invalidate_inference_options_cache() -> None:
    _options_cache.clear()


async def get_inference_option(
    db: Database,
    user_id: str,
    model_id: str,
    *,
    hardware_aware: bool = True,
    profile: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Resolve a single dropdown option without rebuilding the full inventory list."""
    profile = profile if profile is not None else hardware_profile() if hardware_aware else None
    installed = _installed_backends()

    row = await db.get_model(model_id, user_id)
    if not row:
        return None
    return _build_local_option(row, installed=installed, profile=profile)


async def list_inference_options(
    db: Database,
    user_id: str,
    *,
    hardware_aware: bool = True,
    profile: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build dropdown options for chat inference."""
    use_cache = profile is None
    cache_key = (user_id, hardware_aware, id(db))
    if use_cache:
        now = time.monotonic()
        cached = _options_cache.get(cache_key)
        if cached and now - cached[0] < _OPTIONS_CACHE_TTL_S:
            return cached[1]

    profile = profile if profile is not None else hardware_profile() if hardware_aware else None
    installed = _installed_backends()
    options: list[dict[str, Any]] = []

    for row in await db.list_models(user_id):
        opt = _build_local_option(row, installed=installed, profile=profile)
        if opt:
            options.append(opt)

    if profile:
        options.sort(
            key=lambda o: (
                -o.get("hardware_fit_rank", 0),
                o.get("name", ""),
            )
        )

    if use_cache:
        _options_cache[cache_key] = (time.monotonic(), options)
    return options


def resolve_chat_target(
    option: dict[str, Any] | None,
    *,
    model_id: str | None,
    inference_backend: str | None,
) -> dict[str, Any]:
    """Map a dropdown selection to chat payload fields."""
    backend = (inference_backend or "auto").lower()
    if option:
        if backend == "auto":
            backend = option["default_backend"]
        if not backend:
            fmt = (option.get("format") or "").lower()
            if fmt == "gguf":
                raise ValueError(
                    f"{option['name']!r} is a GGUF file, but its architecture is not supported by this llama.cpp runtime. "
                    "Choose another GGUF quant."
                )
            raise ValueError(
                "No installed inference engine can load this model. Install MLX or PyTorch support."
            )
        allowed = set(option.get("backends") or [])
        if allowed and backend not in allowed:
            raise ValueError(
                f"Backend {backend!r} is not available for {option['name']!r}. "
                f"Available: {', '.join(sorted(allowed))}"
            )
        return {
            "model_id": option["id"],
            "model_path": option["path"],
            "inference_backend": backend,
            "model_format": option.get("format"),
        }

    if model_id:
        return {
            "model_id": model_id,
            "model_path": None,
            "inference_backend": backend,
        }
    raise ValueError("Select a model")


def backend_for_path(path: str, fmt: str | None = None) -> str:
    return recommend_backend(model_path=path, model_format=fmt)
