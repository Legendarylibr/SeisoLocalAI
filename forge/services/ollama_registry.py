"""Ollama model registration for downloaded GGUF inventory paths."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from seiso.export.gguf import write_modelfile

logger = logging.getLogger(__name__)

_REGISTRY_FILENAME = "ollama_registry.json"
_TAG_RE = re.compile(r"[^a-z0-9._-]+")


def _data_dir() -> Path:
    from seiso.env import env_str

    raw = env_str("SEISO_DATA_DIR", "~/.seiso").strip() or "~/.seiso"
    return Path(raw).expanduser()


def registry_path() -> Path:
    return _data_dir() / _REGISTRY_FILENAME


def _normalize_key(model_path: str) -> str:
    return str(Path(model_path).expanduser().resolve())


def load_registry() -> dict[str, str]:
    path = registry_path()
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items()}


def save_registry(mapping: dict[str, str]) -> None:
    path = registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(mapping, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def default_ollama_tag(model_path: str, *, repo_id: str | None = None) -> str:
    if repo_id:
        slug = _TAG_RE.sub("-", repo_id.lower().replace("/", "-")).strip("-")
        slug = slug[:96] or "model"
        return f"seiso-{slug}"
    digest = hashlib.sha256(_normalize_key(model_path).encode("utf-8")).hexdigest()[:12]
    return f"seiso-{digest}"


def resolve_ollama_tag(
    model_path: str,
    metadata: dict[str, Any] | None = None,
) -> str:
    key = _normalize_key(model_path)
    registry = load_registry()
    if key in registry:
        return registry[key]
    meta = metadata or {}
    if tag := meta.get("ollama_tag"):
        return str(tag)
    repo_id = meta.get("repo_id")
    if isinstance(repo_id, str) and repo_id:
        return default_ollama_tag(key, repo_id=repo_id)
    return default_ollama_tag(key)


def _ollama_available() -> bool:
    if not shutil.which("ollama"):
        return False
    from seiso.inference.llamaswap import ollama_health_ok, preferred_llamaswap_engine

    return preferred_llamaswap_engine() == "ollama" and ollama_health_ok()


def register_gguf_with_ollama(
    model_path: str,
    *,
    repo_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Create an Ollama model from a local GGUF path and persist the tag mapping."""
    path = Path(model_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"GGUF not found: {path}")

    tag = resolve_ollama_tag(str(path), metadata=metadata)
    if not _ollama_available():
        return tag

    modelfile_dir = path.parent
    write_modelfile(modelfile_dir, path.name)
    modelfile = modelfile_dir / "Modelfile"
    cmd = ["ollama", "create", tag, "-f", str(modelfile)]
    logger.info("Registering GGUF with Ollama: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, timeout=900, capture_output=True, text=True)

    registry = load_registry()
    registry[_normalize_key(str(path))] = tag
    save_registry(registry)
    return tag


def ensure_gguf_registered(
    model_path: str,
    *,
    repo_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Return Ollama tag, registering the GGUF on first use when Ollama is active."""
    key = _normalize_key(model_path)
    registry = load_registry()
    if key in registry:
        return registry[key]
    if not _ollama_available():
        return resolve_ollama_tag(model_path, metadata=metadata)
    try:
        return register_gguf_with_ollama(
            model_path,
            repo_id=repo_id,
            metadata=metadata,
        )
    except Exception as exc:
        logger.warning("Ollama registration failed for %s: %s", model_path, exc)
        return resolve_ollama_tag(model_path, metadata=metadata)
