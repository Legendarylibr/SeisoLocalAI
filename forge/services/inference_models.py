"""Unified inference model list — HF Hub inventory and CLI paths."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from forge.db.store import Database
from forge.services.artifact_integrity import inventory_gguf_is_complete
from forge.services.hardware import (
    assess_inference_option_fit,
    hardware_profile,
)
from forge.services.hf_connectivity import check_inference_runtime
from forge.services.hf_hub import get_gguf_file_size_bytes
from seiso.inference.backends import (
    BACKEND_LABELS,
    resolve_backend_label,
    BACKEND_LLAMACPP,
    BACKEND_LLAMASWAP,
    BACKEND_MLX,
    BACKEND_ROUTER,
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


def _source_label(source: str | None) -> str:
    raw = source or "manual"
    if raw.startswith("hf:"):
        return SOURCE_LABELS["hf:"]
    return SOURCE_LABELS.get(raw, raw)


def _installed_backends() -> dict[str, bool]:
    runtime = check_inference_runtime()
    return {
        BACKEND_LLAMACPP: runtime.llamacpp,
        BACKEND_LLAMASWAP: getattr(runtime, "llamaswap", False),
        BACKEND_MLX: runtime.mlx,
        BACKEND_TORCH: runtime.torch,
    }


def _filter_installed_backends(
    backends: list[str], installed: dict[str, bool]
) -> list[str]:
    return [b for b in backends if installed.get(b, False)]


def _no_backend_status_note(model_format: str | None, install_hints: list[str]) -> str:
    fmt = (model_format or "").lower()
    if fmt == "gguf":
        return (
            install_hints[0]
            if install_hints
            else "GGUF chat requires a reachable isolated backend on this machine. Start Ollama or llama-swap."
        )
    return "No installed inference engine can load this model. Install MLX or PyTorch support."


def _inventory_artifact_is_complete(
    row: dict[str, Any], metadata: dict[str, Any]
) -> bool:
    return inventory_gguf_is_complete(
        row,
        metadata,
        size_lookup=get_gguf_file_size_bytes,
    )


def _enrich_model_runtime_meta(
    opt: dict[str, Any],
    *,
    metadata: dict[str, Any],
) -> None:
    """Attach context/architecture fields once so callers need not re-read GGUF/HF."""
    model_path = opt.get("path")
    model_format = opt.get("format")
    model_name = opt.get("name")
    cached_ceiling = metadata.get("context_ceiling")
    if isinstance(cached_ceiling, int) and cached_ceiling >= 2048:
        opt["context_ceiling"] = cached_ceiling
    else:
        try:
            from seiso.inference.context_limits import effective_context_ceiling

            opt["context_ceiling"] = effective_context_ceiling(
                model_path,
                model_format=model_format,
                model_name=model_name,
            )
        except Exception:
            opt["context_ceiling"] = 8192

    if (model_format or "").lower() == "gguf" and model_path:
        try:
            from seiso.inference.backends import (
                gguf_architecture,
                gguf_is_moe,
                gguf_uses_sliding_window_attention,
            )

            opt["architecture"] = metadata.get("architecture") or gguf_architecture(
                model_path
            )
            opt["is_moe"] = (
                bool(metadata["is_moe"])
                if "is_moe" in metadata
                else gguf_is_moe(model_path)
            )
            opt["uses_swa"] = (
                bool(metadata["uses_swa"])
                if "uses_swa" in metadata
                else gguf_uses_sliding_window_attention(model_path)
            )
        except Exception:
            opt.setdefault("architecture", metadata.get("architecture"))
            opt.setdefault("is_moe", False)
            opt.setdefault("uses_swa", False)
    else:
        opt.setdefault("architecture", metadata.get("architecture"))
        opt.setdefault("is_moe", False)
        opt.setdefault("uses_swa", False)


def _backend_labels_for(backends: list[str]) -> dict[str, str]:
    sidecar_engine = None
    if BACKEND_LLAMASWAP in backends:
        from seiso.inference.sidecar_runtime import sidecar_status

        sidecar_engine = sidecar_status().engine
    return {
        b: resolve_backend_label(b, sidecar_engine=sidecar_engine) for b in backends
    }


def _build_local_option(
    row: dict[str, Any],
    *,
    installed: dict[str, bool],
    profile: dict[str, Any] | None,
) -> dict[str, Any] | None:
    metadata = json.loads(row.get("metadata_json") or "{}")
    complete = _inventory_artifact_is_complete(row, metadata)
    model_format = row.get("format")
    if not complete:
        opt: dict[str, Any] = {
            "id": row["id"],
            "kind": "local",
            "name": row["name"],
            "source": row.get("source") or "manual",
            "source_label": _source_label(row.get("source")),
            "format": model_format,
            "path": row.get("path"),
            "default_backend": "",
            "backends": [],
            "backend_labels": {},
            "size_bytes": row.get("size_bytes", 0),
            "metadata": metadata,
            "selectable": False,
            "status": "incomplete",
            "status_note": "Download incomplete — re-download from Hub to chat with this model.",
            "hardware_note": "Download incomplete — re-download from Hub to chat with this model.",
            "hardware_fit": "unlikely",
            "hardware_fit_label": "Incomplete",
            "hardware_fit_rank": -1,
            "memory_load_blocked": True,
            "memory_load_blocked_reason": "Download incomplete — re-download from Hub.",
        }
        from forge.services.inference_variants import extract_quant_label, variant_group_key

        opt["quant_label"] = extract_quant_label(
            name=row["name"],
            path=row.get("path") or "",
            metadata=metadata,
        )
        opt["variant_group"] = variant_group_key(opt)
        return opt

    candidate_backends = available_backends(
        model_path=row["path"],
        model_format=model_format,
    )
    backends = _filter_installed_backends(
        candidate_backends,
        installed,
    )
    opt = {
        "id": row["id"],
        "kind": "local",
        "name": row["name"],
        "source": row.get("source") or "manual",
        "source_label": _source_label(row.get("source")),
        "format": model_format,
        "path": row["path"],
        "default_backend": backends[0] if backends else "",
        "backends": backends,
        "backend_labels": _backend_labels_for(backends),
        "size_bytes": row.get("size_bytes", 0),
        "metadata": metadata,
        "selectable": True,
        "status": "ready",
    }
    from forge.services.inference_variants import extract_quant_label, variant_group_key

    opt["quant_label"] = extract_quant_label(
        name=row["name"],
        path=row["path"],
        metadata=metadata,
    )
    opt["variant_group"] = variant_group_key(opt)
    _enrich_model_runtime_meta(opt, metadata=metadata)
    if not backends and (row.get("format") or "").lower() == "gguf":
        runtime = check_inference_runtime()
        opt["install_hints"] = [
            hint for hint in runtime.install_hints if "llama" in hint.lower()
        ] or ["Start Ollama or llama-swap for GGUF chat"]
    if not backends:
        note = _no_backend_status_note(model_format, opt.get("install_hints") or [])
        opt.update(
            {
                "selectable": False,
                "status": "unavailable",
                "status_note": note,
                "hardware_note": note,
                "memory_load_blocked": True,
                "memory_load_blocked_reason": note,
            }
        )
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
    profile = (
        profile
        if profile is not None
        else hardware_profile() if hardware_aware else None
    )
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
    model_router_enabled: bool = False,
) -> list[dict[str, Any]]:
    """Build dropdown options for chat inference."""
    use_cache = profile is None
    cache_key = (user_id, hardware_aware, model_router_enabled, id(db))
    if use_cache:
        now = time.monotonic()
        cached = _options_cache.get(cache_key)
        if cached and now - cached[0] < _OPTIONS_CACHE_TTL_S:
            return cached[1]

    profile = (
        profile
        if profile is not None
        else hardware_profile() if hardware_aware else None
    )
    installed = _installed_backends()
    options: list[dict[str, Any]] = []

    if model_router_enabled:
        options.append(
            {
                "id": "__seiso_router__",
                "kind": "router",
                "name": "Smart Router (auto-route)",
                "source": "router",
                "source_label": "External router",
                "format": None,
                "path": None,
                "default_backend": BACKEND_ROUTER,
                "backends": [BACKEND_ROUTER],
                "backend_labels": {BACKEND_ROUTER: BACKEND_LABELS[BACKEND_ROUTER]},
                "size_bytes": 0,
                "metadata": {"description": "External local router service"},
                "selectable": True,
                "status": "ready",
                "hardware_fit": "ideal",
                "hardware_fit_label": "Managed routing",
                "hardware_fit_rank": 100,
            }
        )

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
    if option and option.get("kind") == "router":
        return {
            "model_id": option["id"],
            "model_path": None,
            "inference_backend": BACKEND_ROUTER,
        }

    backend = (inference_backend or "auto").lower()
    if option:
        if backend == "auto":
            backend = option["default_backend"]
        if not backend:
            raise ValueError(
                option.get("status_note")
                or option.get("hardware_note")
                or _no_backend_status_note(option.get("format"), option.get("install_hints") or [])
            )
        allowed = set(option.get("backends") or [])
        if not allowed:
            raise ValueError(
                option.get("status_note")
                or option.get("hardware_note")
                or _no_backend_status_note(option.get("format"), option.get("install_hints") or [])
            )
        if backend not in allowed:
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
        raise ValueError("Model not found in inventory")
    raise ValueError("Select a model")


def backend_for_path(path: str, fmt: str | None = None) -> str:
    return recommend_backend(model_path=path, model_format=fmt)
