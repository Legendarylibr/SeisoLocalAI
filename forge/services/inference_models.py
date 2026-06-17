"""Unified inference model list — HF Hub inventory, CLI paths, Ollama tags."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from forge.db.store import Database
from forge.providers.ollama import list_models as list_ollama_models
from seiso.inference.backends import (
    BACKEND_OLLAMA,
    available_backends,
    match_ollama_name,
    recommend_backend,
)

logger = logging.getLogger(__name__)

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


def _source_label(source: str | None) -> str:
    raw = source or "manual"
    if raw.startswith("hf:"):
        return SOURCE_LABELS["hf:"]
    return SOURCE_LABELS.get(raw, raw)


async def _ollama_names(base_url: str = "") -> set[str]:
    try:
        models = await list_ollama_models(base_url)
        return {m["name"] for m in models if m.get("name")}
    except Exception as exc:
        logger.debug("Ollama unavailable: %s", exc)
        return set()


async def list_inference_options(
    db: Database,
    user_id: str,
    *,
    ollama_base_url: str = "",
) -> list[dict[str, Any]]:
    """Build dropdown options for chat inference."""
    ollama_tags = await _ollama_names(ollama_base_url)
    options: list[dict[str, Any]] = []

    for row in await db.list_models(user_id):
        metadata = json.loads(row.get("metadata_json") or "{}")
        backends = available_backends(
            model_path=row["path"],
            model_format=row.get("format"),
            ollama_names=ollama_tags,
        )
        ollama_match = match_ollama_name(
            model_path=row["path"],
            model_name=row["name"],
            ollama_names=ollama_tags,
        )
        options.append(
            {
                "id": row["id"],
                "kind": "local",
                "name": row["name"],
                "source": row.get("source") or "manual",
                "source_label": _source_label(row.get("source")),
                "format": row.get("format"),
                "path": row["path"],
                "default_backend": backends[0],
                "backends": backends,
                "backend_labels": {b: BACKEND_LABELS.get(b, b) for b in backends},
                "ollama_model": ollama_match,
                "size_bytes": row.get("size_bytes", 0),
                "metadata": metadata,
            }
        )

    seen_ollama: set[str] = set()
    for opt in options:
        if opt.get("ollama_model"):
            seen_ollama.add(opt["ollama_model"])

    for tag in sorted(ollama_tags):
        if tag in seen_ollama:
            continue
        options.append(
            {
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
        )

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
