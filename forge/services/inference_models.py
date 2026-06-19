"""Unified inference model list — HF Hub inventory, CLI paths, Ollama tags."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from forge.db.store import Database
from forge.providers.ollama import list_models as list_ollama_models
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
    BACKEND_LLAMACPP,
    BACKEND_MLX,
    BACKEND_OLLAMA,
    BACKEND_TORCH,
    available_backends,
    gguf_is_supported_by_llamacpp,
    match_ollama_name,
    recommend_backend,
)

logger = logging.getLogger(__name__)

_OLLAMA_CACHE_TTL_S = 10.0
_OPTIONS_CACHE_TTL_S = 5.0
_ollama_cache: set[str] | None = None
_ollama_cache_key: str = ""
_ollama_cache_ts: float = 0.0
_options_cache: dict[tuple, tuple[float, list[dict[str, Any]]]] = {}

BACKEND_LABELS = {
    "llamacpp": "llama.cpp",
    "ollama": "Ollama",
    "mlx": "MLX",
    "torch": "PyTorch",
    "auto": "Auto",
}

SOURCE_LABELS = {
    "hf:": "Hugging Face Hub",
    "scan": "CLI scan",
    "manual": "CLI path",
    "training": "Fine-tune output",
    "export": "Export output",
    "ollama": "Ollama",
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
        if BACKEND_OLLAMA in backends and BACKEND_LLAMACPP in backends:
            tier = classify_tier(profile)
            if tier in (HardwareTier.CPU_ONLY, HardwareTier.EDGE):
                return BACKEND_LLAMACPP
            if vram_headroom_mb(profile) >= 8000:
                return BACKEND_OLLAMA
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
        BACKEND_OLLAMA: True,
    }


def _filter_installed_backends(backends: list[str], installed: dict[str, bool]) -> list[str]:
    return [b for b in backends if installed.get(b, False)]


def _inventory_artifact_is_complete(row: dict[str, Any], metadata: dict[str, Any]) -> bool:
    path = Path(str(row.get("path") or ""))
    if not path.exists():
        return False
    fmt = str(row.get("format") or "").lower()
    if fmt != "gguf":
        return True
    if not gguf_is_supported_by_llamacpp(str(path)):
        return False

    gguf_repo = str(metadata.get("gguf_repo") or metadata.get("repo_id") or "")
    gguf_files = metadata.get("gguf_files") or metadata.get("gguf_file")
    if isinstance(gguf_files, str):
        gguf_files = [gguf_files]
    if not gguf_repo or not isinstance(gguf_files, list) or not gguf_files:
        return not str(row.get("source") or "").startswith("hf:")

    local_files = [path] if path.is_file() else [path / str(filename) for filename in gguf_files]
    if not all(item.is_file() for item in local_files):
        return False
    actual_size = sum(item.stat().st_size for item in local_files)
    try:
        expected_size = sum(get_gguf_file_size_bytes(gguf_repo, str(filename)) for filename in gguf_files)
    except Exception:
        return True
    return expected_size <= 0 or actual_size >= expected_size


def _build_local_option(
    row: dict[str, Any],
    *,
    installed: dict[str, bool],
    ollama_tags: set[str],
    profile: dict[str, Any] | None,
) -> dict[str, Any] | None:
    metadata = json.loads(row.get("metadata_json") or "{}")
    if not _inventory_artifact_is_complete(row, metadata):
        return None
    backends = _filter_installed_backends(
        available_backends(
            model_path=row["path"],
            model_format=row.get("format"),
            ollama_names=ollama_tags,
        ),
        installed,
    )
    ollama_match = match_ollama_name(
        model_path=row["path"],
        model_name=row["name"],
        ollama_names=ollama_tags,
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
        "ollama_model": ollama_match,
        "size_bytes": row.get("size_bytes", 0),
        "metadata": metadata,
    }
    if profile:
        opt.update(assess_inference_option_fit(opt, profile))
    return opt


def _build_ollama_option(
    tag: str,
    *,
    profile: dict[str, Any] | None,
) -> dict[str, Any]:
    opt = {
        "id": f"ollama:{tag}",
        "kind": "ollama",
        "name": tag,
        "source": "ollama",
        "source_label": SOURCE_LABELS["ollama"],
        "format": None,
        "path": None,
        "default_backend": BACKEND_OLLAMA,
        "backends": [BACKEND_OLLAMA],
        "backend_labels": {BACKEND_OLLAMA: BACKEND_LABELS[BACKEND_OLLAMA]},
        "ollama_model": tag,
        "size_bytes": 0,
        "metadata": {},
    }
    if profile:
        opt.update(assess_inference_option_fit(opt, profile))
    return opt


def invalidate_inference_options_cache() -> None:
    _options_cache.clear()


async def _ollama_names(base_url: str = "") -> set[str]:
    global _ollama_cache, _ollama_cache_key, _ollama_cache_ts

    now = time.time()
    if (
        _ollama_cache is not None
        and _ollama_cache_key == base_url
        and now - _ollama_cache_ts < _OLLAMA_CACHE_TTL_S
    ):
        return _ollama_cache

    try:
        models = await list_ollama_models(base_url)
        result = {m["name"] for m in models if m.get("name")}
    except Exception as exc:
        logger.debug("Ollama unavailable: %s", exc)
        result = set()

    _ollama_cache = result
    _ollama_cache_key = base_url
    _ollama_cache_ts = now
    return result


async def get_inference_option(
    db: Database,
    user_id: str,
    model_id: str,
    *,
    ollama_base_url: str = "",
    hardware_aware: bool = True,
    profile: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Resolve a single dropdown option without rebuilding the full inventory list."""
    profile = profile if profile is not None else hardware_profile() if hardware_aware else None
    installed = _installed_backends()
    ollama_tags = await _ollama_names(ollama_base_url)

    if model_id.startswith("ollama:"):
        tag = model_id.removeprefix("ollama:")
        if tag not in ollama_tags:
            return None
        return _build_ollama_option(tag, profile=profile)

    row = await db.get_model(model_id, user_id)
    if not row:
        return None
    return _build_local_option(row, installed=installed, ollama_tags=ollama_tags, profile=profile)


async def list_inference_options(
    db: Database,
    user_id: str,
    *,
    ollama_base_url: str = "",
    hardware_aware: bool = True,
    profile: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build dropdown options for chat inference."""
    use_cache = profile is None
    cache_key = (user_id, ollama_base_url, hardware_aware, id(db))
    if use_cache:
        now = time.monotonic()
        cached = _options_cache.get(cache_key)
        if cached and now - cached[0] < _OPTIONS_CACHE_TTL_S:
            return cached[1]

    profile = profile if profile is not None else hardware_profile() if hardware_aware else None
    installed = _installed_backends()
    ollama_tags = await _ollama_names(ollama_base_url)
    options: list[dict[str, Any]] = []

    for row in await db.list_models(user_id):
        opt = _build_local_option(row, installed=installed, ollama_tags=ollama_tags, profile=profile)
        if opt:
            options.append(opt)

    seen_ollama: set[str] = set()
    for opt in options:
        if opt.get("ollama_model"):
            seen_ollama.add(opt["ollama_model"])

    for tag in sorted(ollama_tags):
        if tag in seen_ollama:
            continue
        options.append(_build_ollama_option(tag, profile=profile))

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
    ollama_model: str | None,
    inference_backend: str | None,
) -> dict[str, Any]:
    """Map a dropdown selection to chat payload fields."""
    if option and option.get("kind") == "ollama":
        return {
            "model_id": None,
            "model_path": None,
            "ollama_model": option["ollama_model"],
            "inference_backend": BACKEND_OLLAMA,
        }

    backend = (inference_backend or "auto").lower()
    if option:
        if backend == "auto":
            backend = option["default_backend"]
        if not backend:
            fmt = (option.get("format") or "").lower()
            if fmt == "gguf":
                raise ValueError(
                    f"{option['name']!r} is a GGUF file, but its architecture is not supported by this llama.cpp runtime. "
                    "Choose another GGUF quant or an Ollama model."
                )
            raise ValueError("No installed inference engine can load this model. Install MLX or PyTorch support.")
        if backend == BACKEND_OLLAMA:
            tag = option.get("ollama_model")
            if not tag:
                raise ValueError(
                    f"Model {option['name']!r} is not available in Ollama — use llama.cpp or import with ollama create"
                )
            return {
                "model_id": option["id"],
                "model_path": option["path"],
                "ollama_model": tag,
                "inference_backend": BACKEND_OLLAMA,
                "model_format": option.get("format"),
            }
        allowed = set(option.get("backends") or [])
        if allowed and backend not in allowed:
            raise ValueError(
                f"Backend {backend!r} is not available for {option['name']!r}. "
                f"Available: {', '.join(sorted(allowed))}"
            )
        return {
            "model_id": option["id"],
            "model_path": option["path"],
            "ollama_model": None,
            "inference_backend": backend,
            "model_format": option.get("format"),
        }

    if ollama_model:
        return {
            "model_id": None,
            "model_path": None,
            "ollama_model": ollama_model,
            "inference_backend": BACKEND_OLLAMA,
        }
    if model_id:
        return {
            "model_id": model_id,
            "model_path": None,
            "ollama_model": None,
            "inference_backend": backend,
        }
    raise ValueError("Select a model")


def backend_for_path(path: str, fmt: str | None = None) -> str:
    return recommend_backend(model_path=path, model_format=fmt)
