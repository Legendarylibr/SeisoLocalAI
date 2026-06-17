"""Discover local llama.cpp binaries and GGUF models for the launcher dashboard."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from adaptive_quant.configuration.sections import (
    default_route_catalog_path,
    default_route_models_dir,
)
from adaptive_quant.logging_utils import read_json
from adaptive_quant.routing import parse_route

_LLAMA_BINARY_NAMES = ("llama-cli", "llama-server", "main", "llama")
_ENV_BINARY = "LLAMA_CPP_BINARY"
_ENV_MODEL = "LLAMA_CPP_MODEL"


@dataclass(frozen=True)
class LocalModelEntry:
    id: str
    label: str
    model_path: str
    source: str
    route_id: str | None = None
    quant_label: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "model_path": self.model_path,
            "source": self.source,
            "route_id": self.route_id,
            "quant_label": self.quant_label,
        }


def _model_id(model_path: str) -> str:
    digest = hashlib.sha256(model_path.encode("utf-8")).hexdigest()[:16]
    name = Path(model_path).name
    return f"{name}:{digest}"


def _is_executable(path: Path) -> bool:
    return path.is_file() and os.access(path, os.X_OK)


def _is_gguf(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() == ".gguf"


def _resolve_existing(path: str) -> Path | None:
    candidate = Path(path).expanduser()
    try:
        resolved = candidate.resolve()
    except OSError:
        return None
    return resolved if resolved.is_file() else None


def discover_llama_cpp_binary(*, repo: Path) -> str | None:
    """Return the first usable llama.cpp-class binary path, or None."""
    seen: set[str] = set()

    def accept(path: Path) -> str | None:
        if not _is_executable(path):
            return None
        key = str(path.resolve())
        if key in seen:
            return None
        seen.add(key)
        return key

    env_binary = os.environ.get(_ENV_BINARY, "").strip()
    if env_binary:
        resolved = _resolve_existing(env_binary)
        if resolved is not None:
            accepted = accept(resolved)
            if accepted:
                return accepted

    for config_path in sorted(repo.glob("config*.json")):
        if not config_path.is_file():
            continue
        try:
            payload = read_json(str(config_path))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        binary_raw = payload.get("llama_cpp_binary")
        if not isinstance(binary_raw, str) or not binary_raw.strip():
            nested = payload.get("llama_cpp")
            if isinstance(nested, dict):
                binary_raw = nested.get("binary")
        if not isinstance(binary_raw, str) or not binary_raw.strip():
            continue
        resolved = _resolve_existing(binary_raw.strip())
        if resolved is not None:
            accepted = accept(resolved)
            if accepted:
                return accepted

    for name in _LLAMA_BINARY_NAMES:
        found = shutil.which(name)
        if not found:
            continue
        accepted = accept(Path(found))
        if accepted:
            return accepted

    return None


def _add_model(
    entries: dict[str, LocalModelEntry],
    *,
    model_path: str,
    label: str,
    source: str,
    route_id: str | None = None,
    quant_label: str | None = None,
) -> None:
    resolved = _resolve_existing(model_path)
    if resolved is None or not _is_gguf(resolved):
        return
    path_str = str(resolved)
    entry_id = _model_id(path_str)
    if entry_id in entries:
        return
    entries[entry_id] = LocalModelEntry(
        id=entry_id,
        label=label,
        model_path=path_str,
        source=source,
        route_id=route_id,
        quant_label=quant_label,
    )


def _models_from_config_files(repo: Path, entries: dict[str, LocalModelEntry]) -> None:
    for config_path in sorted(repo.glob("config*.json")):
        if not config_path.is_file():
            continue
        try:
            payload = read_json(str(config_path))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        model_raw = payload.get("llama_cpp_model")
        if not isinstance(model_raw, str) or not model_raw.strip():
            nested = payload.get("llama_cpp")
            if isinstance(nested, dict):
                model_raw = nested.get("model")
        if isinstance(model_raw, str) and model_raw.strip():
            _add_model(
                entries,
                model_path=model_raw.strip(),
                label=f"{Path(model_raw.strip()).name} ({config_path.name})",
                source="config",
            )
        routes = payload.get("router_routes")
        if isinstance(routes, list):
            for route in routes:
                if not isinstance(route, str):
                    continue
                try:
                    candidate = parse_route(route)
                except (ValueError, TypeError):
                    continue
                if candidate.backend != "llama_cpp":
                    continue
                _add_model(
                    entries,
                    model_path=candidate.model_id,
                    label=f"{Path(candidate.model_id).name} (route)",
                    source="router_route",
                )


def _models_from_route_catalog(repo: Path, entries: dict[str, LocalModelEntry]) -> None:
    catalog_path = repo / default_route_catalog_path("outputs")
    if not catalog_path.is_file():
        return
    try:
        payload = read_json(str(catalog_path))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return
    routes = payload.get("routes") if isinstance(payload, dict) else None
    if not isinstance(routes, list):
        return
    for row in routes:
        if not isinstance(row, dict):
            continue
        local_path = row.get("local_path")
        if not isinstance(local_path, str) or not local_path.strip():
            continue
        route_id = str(row.get("route_id") or Path(local_path).stem)
        quant = row.get("quant_label")
        quant_label = str(quant) if quant is not None else None
        label = route_id
        if quant_label:
            label = f"{route_id} ({quant_label})"
        _add_model(
            entries,
            model_path=local_path.strip(),
            label=label,
            source="route_catalog",
            route_id=route_id,
            quant_label=quant_label,
        )


def _models_from_scan(repo: Path, entries: dict[str, LocalModelEntry]) -> None:
    models_dir = repo / default_route_models_dir("outputs")
    if not models_dir.is_dir():
        return
    for path in sorted(models_dir.rglob("*.gguf")):
        if not _is_gguf(path):
            continue
        rel = path.relative_to(repo)
        _add_model(
            entries,
            model_path=str(path),
            label=f"{path.name} ({rel.parent})",
            source="outputs_models",
        )


def discover_local_models(*, repo: Path) -> list[LocalModelEntry]:
    """Collect discoverable local GGUF models under the repo."""
    entries: dict[str, LocalModelEntry] = {}

    env_model = os.environ.get(_ENV_MODEL, "").strip()
    if env_model:
        _add_model(
            entries,
            model_path=env_model,
            label=f"{Path(env_model).name} (env)",
            source="env",
        )

    _models_from_config_files(repo, entries)
    _models_from_route_catalog(repo, entries)
    _models_from_scan(repo, entries)

    return sorted(entries.values(), key=lambda item: item.label.lower())


def local_model_catalog(*, repo: Path) -> dict[str, Any]:
    """Payload for launcher UI model selection."""
    binary = discover_llama_cpp_binary(repo=repo)
    models = discover_local_models(repo=repo)
    selected_path = _load_selected_model_path(repo)
    selected_id = None
    if selected_path:
        selected_id = _model_id(selected_path)
    model_ids = {item.id for item in models}
    if selected_id and selected_id not in model_ids:
        models = sorted(
            [
                *models,
                LocalModelEntry(
                    id=selected_id,
                    label=Path(selected_path).name,
                    model_path=selected_path,
                    source="session",
                ),
            ],
            key=lambda item: item.label.lower(),
        )
    return {
        "llama_cpp_binary": binary,
        "models": [item.to_dict() for item in models],
        "selected_model_id": selected_id,
        "llama_ready": bool(binary and models),
    }


def resolve_local_model(
    *,
    repo: Path,
    model_id: str | None,
) -> LocalModelEntry | None:
    if not model_id or not str(model_id).strip():
        return None
    target = str(model_id).strip()
    for item in discover_local_models(repo=repo):
        if item.id == target:
            return item
    selected_path = _load_selected_model_path(repo)
    if selected_path and _model_id(selected_path) == target:
        return LocalModelEntry(
            id=target,
            label=Path(selected_path).name,
            model_path=selected_path,
            source="session",
        )
    return None


def _session_config_path(repo: Path) -> Path:
    return repo / "outputs" / ".launcher_chat_session" / "session_config.json"


def _load_selected_model_path(repo: Path) -> str | None:
    path = _session_config_path(repo)
    if not path.is_file():
        return None
    try:
        payload = read_json(str(path))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    model = payload.get("llama_cpp_model")
    if isinstance(model, str) and model.strip():
        return model.strip()
    nested = payload.get("llama_cpp")
    if isinstance(nested, dict):
        nested_model = nested.get("model")
        if isinstance(nested_model, str) and nested_model.strip():
            return nested_model.strip()
    return None


__all__ = [
    "LocalModelEntry",
    "discover_llama_cpp_binary",
    "discover_local_models",
    "local_model_catalog",
    "resolve_local_model",
]
