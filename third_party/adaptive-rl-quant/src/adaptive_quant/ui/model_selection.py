"""Launcher model selection: local GGUF files, Hugging Face route catalog, and downloads."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from adaptive_quant.configuration.sections import (
    default_route_catalog_path,
    default_route_models_dir,
)
from adaptive_quant.configuration.validation import hf_allowed_repos_from_env
from adaptive_quant.huggingface_cli import (
    find_huggingface_cli,
    require_huggingface_cli,
    run_download,
)
from adaptive_quant.logging_utils import read_json, write_json
from adaptive_quant.model_routes import ModelRoute, RouteCatalog, default_route_catalog

_ENV_BINARY = "LLAMA_CPP_BINARY"
_ENV_MODEL = "LLAMA_CPP_MODEL"
_HF_API = "https://huggingface.co/api"
_LLAMA_BINARY_NAMES = ("llama-cli", "llama-server", "main", "llama")
_SELECTION_STATE = "outputs/.launcher_model_selection.json"


@dataclass(frozen=True)
class ModelChoice:
    id: str
    label: str
    source: str
    ready: bool
    model_path: str | None = None
    route_id: str | None = None
    repo_id: str | None = None
    filename: str | None = None
    quant_label: str | None = None
    size_mb: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "source": self.source,
            "ready": self.ready,
            "model_path": self.model_path,
            "route_id": self.route_id,
            "repo_id": self.repo_id,
            "filename": self.filename,
            "quant_label": self.quant_label,
            "size_mb": self.size_mb,
        }


def _selection_state_path(repo: Path) -> Path:
    return repo / _SELECTION_STATE


def load_selected_model_id(repo: Path) -> str | None:
    path = _selection_state_path(repo)
    if not path.is_file():
        return None
    try:
        payload = read_json(str(path))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    selected = payload.get("selected_model_id")
    return str(selected).strip() if isinstance(selected, str) and selected.strip() else None


def save_selected_model_id(repo: Path, model_id: str | None) -> None:
    path = _selection_state_path(repo)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(str(path), {"selected_model_id": model_id})


def _is_executable(path: Path) -> bool:
    return path.is_file() and os.access(path, os.X_OK)


def _is_gguf(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() == ".gguf"


def _resolve_file(path: str) -> Path | None:
    candidate = Path(path).expanduser()
    try:
        resolved = candidate.resolve()
    except OSError:
        return None
    return resolved if resolved.is_file() else None


def discover_llama_cpp_binary(*, repo: Path) -> str | None:
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
        resolved = _resolve_file(env_binary)
        if resolved is not None:
            accepted = accept(resolved)
            if accepted:
                return accepted

    for config_path in sorted(repo.glob("config*.json")):
        try:
            payload = read_json(str(config_path))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        binary_raw = payload.get("llama_cpp_binary")
        if not isinstance(binary_raw, str) and isinstance(payload.get("llama_cpp"), dict):
            binary_raw = payload["llama_cpp"].get("binary")
        if isinstance(binary_raw, str) and binary_raw.strip():
            resolved = _resolve_file(binary_raw.strip())
            if resolved is not None:
                accepted = accept(resolved)
                if accepted:
                    return accepted

    for name in _LLAMA_BINARY_NAMES:
        found = shutil.which(name)
        if found:
            accepted = accept(Path(found))
            if accepted:
                return accepted
    return None


def _route_model_path(route: ModelRoute) -> str | None:
    if route.local_path:
        resolved = _resolve_file(route.local_path)
        if resolved is not None and _is_gguf(resolved):
            return str(resolved)
        base = Path(route.local_path).expanduser()
        if base.is_dir() and route.filename:
            candidate = base / route.filename
            if _is_gguf(candidate):
                return str(candidate.resolve())
    if route.filename:
        models_root = Path(default_route_models_dir())
        candidate = models_root / route.route_id / route.filename
        if _is_gguf(candidate):
            return str(candidate.resolve())
    return None


def _local_id(model_path: str) -> str:
    digest = hashlib.sha256(model_path.encode("utf-8")).hexdigest()[:12]
    return f"local:{Path(model_path).name}:{digest}"


def _route_id(route: ModelRoute) -> str:
    return f"route:{route.route_id}"


def load_route_catalog(repo: Path) -> RouteCatalog:
    catalog_path = repo / default_route_catalog_path("outputs")
    if catalog_path.is_file():
        try:
            return RouteCatalog.from_file(catalog_path)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass
    catalog = default_route_catalog()
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog.save(catalog_path)
    return catalog


def list_model_choices(*, repo: Path) -> list[ModelChoice]:
    choices: dict[str, ModelChoice] = {}

    env_model = os.environ.get(_ENV_MODEL, "").strip()
    if env_model:
        resolved = _resolve_file(env_model)
        if resolved is not None and _is_gguf(resolved):
            path_str = str(resolved)
            choices[_local_id(path_str)] = ModelChoice(
                id=_local_id(path_str),
                label=f"{resolved.name} (env)",
                source="env",
                ready=True,
                model_path=path_str,
            )

    for config_path in sorted(repo.glob("config*.json")):
        try:
            payload = read_json(str(config_path))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        model_raw = payload.get("llama_cpp_model")
        if not isinstance(model_raw, str) and isinstance(payload.get("llama_cpp"), dict):
            model_raw = payload["llama_cpp"].get("model")
        if isinstance(model_raw, str) and model_raw.strip():
            resolved = _resolve_file(model_raw.strip())
            if resolved is not None and _is_gguf(resolved):
                path_str = str(resolved)
                choices[_local_id(path_str)] = ModelChoice(
                    id=_local_id(path_str),
                    label=f"{resolved.name} ({config_path.name})",
                    source="config",
                    ready=True,
                    model_path=path_str,
                )

    models_dir = repo / default_route_models_dir("outputs")
    if models_dir.is_dir():
        for path in sorted(models_dir.rglob("*.gguf")):
            if not _is_gguf(path):
                continue
            path_str = str(path.resolve())
            choices[_local_id(path_str)] = ModelChoice(
                id=_local_id(path_str),
                label=f"{path.name} ({path.parent.name})",
                source="local",
                ready=True,
                model_path=path_str,
            )

    catalog = load_route_catalog(repo)
    for route in catalog.routes:
        model_path = _route_model_path(route)
        entry_id = _route_id(route)
        label = route.route_id
        if route.quant_label:
            label = f"{route.route_id} · {route.quant_label}"
        if route.repo_id:
            label = f"{label} · {route.repo_id}"
        choices[entry_id] = ModelChoice(
            id=entry_id,
            label=label,
            source="huggingface_route",
            ready=model_path is not None,
            model_path=model_path,
            route_id=route.route_id,
            repo_id=route.repo_id,
            filename=route.filename,
            quant_label=route.quant_label,
            size_mb=route.size_mb,
        )

    return sorted(choices.values(), key=lambda item: (not item.ready, item.label.lower()))


def resolve_model_choice(*, repo: Path, model_id: str | None) -> ModelChoice | None:
    if not model_id or not str(model_id).strip():
        return None
    target = str(model_id).strip()
    for choice in list_model_choices(repo=repo):
        if choice.id == target:
            return choice
    return None


def search_huggingface_gguf_repos(*, query: str, limit: int = 12) -> list[dict[str, Any]]:
    """Search Hugging Face Hub for GGUF model repos (stdlib HTTP, no token required)."""
    q = query.strip()
    if not q:
        return []
    params = urllib.parse.urlencode(
        {
            "search": q,
            "filter": "gguf",
            "limit": max(1, min(limit, 25)),
            "full": "false",
        }
    )
    url = f"{_HF_API}/models?{params}"
    request = urllib.request.Request(url, headers={"User-Agent": "adaptive-rl-quant-launcher/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=15.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return []
    if not isinstance(payload, list):
        return []
    results: list[dict[str, Any]] = []
    for row in payload:
        if not isinstance(row, dict):
            continue
        model_id = row.get("id") or row.get("modelId")
        if not isinstance(model_id, str):
            continue
        results.append(
            {
                "repo_id": model_id,
                "downloads": row.get("downloads"),
                "likes": row.get("likes"),
                "tags": row.get("tags") if isinstance(row.get("tags"), list) else [],
            }
        )
    return results


def download_route_model(
    *,
    repo: Path,
    route_id: str,
    timeout_s: float = 600.0,
) -> ModelChoice:
    """Download a Hugging Face route catalog entry via ``hf download``."""
    catalog_path = repo / default_route_catalog_path("outputs")
    catalog = load_route_catalog(repo)
    route = catalog.by_id(route_id)
    cli = require_huggingface_cli()
    local_dir = repo / default_route_models_dir("outputs") / route.route_id
    local_dir.mkdir(parents=True, exist_ok=True)
    allowed_repos = tuple(hf_allowed_repos_from_env())
    result = run_download(
        cli,
        repo_id=route.repo_id,
        filename=route.filename,
        revision=route.revision,
        local_dir=local_dir,
        allowed_repos=allowed_repos,
        timeout_s=timeout_s,
    )
    if result.timed_out:
        raise TimeoutError(f"Hugging Face download timed out after {timeout_s}s")
    if not result.ok:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(
            f"Hugging Face download failed (exit {result.returncode}). {detail}".strip()
        )

    resolved = result.local_path
    if resolved is None and route.filename:
        candidate = local_dir / route.filename
        if candidate.is_file():
            resolved = candidate
    if resolved is None:
        ggufs = sorted(local_dir.rglob("*.gguf"))
        if ggufs:
            resolved = ggufs[0]
    if resolved is None:
        resolved = local_dir

    model_path = str(resolved.resolve()) if resolved.is_file() else None
    if model_path is None and resolved.is_dir() and route.filename:
        candidate = resolved / route.filename
        if candidate.is_file():
            model_path = str(candidate.resolve())

    catalog.update_local_path(route.route_id, model_path or str(resolved))
    catalog.save(catalog_path)

    choice = resolve_model_choice(repo=repo, model_id=_route_id(route))
    if choice is None:
        raise RuntimeError(f"Download succeeded but route {route_id!r} is not listed")
    if not choice.ready:
        raise RuntimeError(
            f"Download finished but no GGUF file was found for route {route_id!r} under {local_dir}"
        )
    return choice


def model_catalog_payload(*, repo: Path) -> dict[str, Any]:
    binary = discover_llama_cpp_binary(repo=repo)
    models = list_model_choices(repo=repo)
    selected = load_selected_model_id(repo)
    hf_cli = find_huggingface_cli()
    return {
        "llama_cpp_binary": binary,
        "llama_ready": bool(binary and any(item.ready for item in models)),
        "models": [item.to_dict() for item in models],
        "selected_model_id": selected,
        "huggingface_cli": hf_cli.binary if hf_cli else None,
        "route_catalog_path": default_route_catalog_path("outputs"),
    }


__all__ = [
    "ModelChoice",
    "discover_llama_cpp_binary",
    "download_route_model",
    "list_model_choices",
    "load_selected_model_id",
    "model_catalog_payload",
    "resolve_model_choice",
    "save_selected_model_id",
    "search_huggingface_gguf_repos",
]
