"""Ollama model registration for local inventory paths (GGUF and reference mappings)."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from seiso.export.gguf import write_modelfile

logger = logging.getLogger(__name__)

_REGISTRY_FILENAME = "ollama_registry.json"
_TAG_RE = re.compile(r"[^a-z0-9._-]+")
_MAX_TAG_LEN = 120


@dataclass(frozen=True)
class OllamaArtifact:
    kind: str  # gguf | reference | pull
    path: Path | None = None
    tag: str | None = None
    pull_name: str | None = None


def _data_dir() -> Path:
    from seiso.env import env_str

    raw = env_str("SEISO_DATA_DIR", "~/.seiso").strip() or "~/.seiso"
    return Path(raw).expanduser()


def registry_path() -> Path:
    return _data_dir() / _REGISTRY_FILENAME


def _normalize_key(model_path: str) -> str:
    return str(Path(model_path).expanduser().resolve())


def _slug(value: str, *, max_len: int = 48) -> str:
    slug = _TAG_RE.sub("-", value.lower()).replace("_", "-").strip("-")
    return slug[:max_len] or "model"


def _merge_metadata(
    metadata: dict[str, Any] | None,
    *,
    model_path: str | None = None,
    model_format: str | None = None,
) -> dict[str, Any]:
    meta = dict(metadata or {})
    if model_path and not meta.get("gguf_file"):
        path = Path(model_path)
        if path.is_file():
            meta.setdefault("gguf_file", path.name)
        elif path.is_dir():
            meta.setdefault("inventory_path", str(path))
    if model_format and not meta.get("format"):
        meta["format"] = model_format
    return meta


def _quant_slug(metadata: dict[str, Any], model_path: str) -> str:
    gguf_file = metadata.get("gguf_file")
    if isinstance(gguf_file, str) and gguf_file.strip():
        return _slug(Path(gguf_file).stem)
    path = Path(model_path)
    if path.is_file() or path.suffix.lower() == ".gguf":
        return _slug(path.stem)
    if path.is_dir():
        try:
            from seiso.inference.backends import resolve_gguf_file

            return _slug(resolve_gguf_file(str(path)).stem)
        except Exception:
            pass
    return _slug(path.name or "model")


def default_ollama_tag(
    model_path: str,
    *,
    repo_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    meta = _merge_metadata(metadata, model_path=model_path)
    if explicit := meta.get("ollama_tag"):
        return _slug(str(explicit), max_len=_MAX_TAG_LEN)
    if pull := meta.get("ollama_pull"):
        pull_name = str(pull).strip()
        if pull_name:
            return _slug(pull_name.split(":")[0], max_len=_MAX_TAG_LEN)
    quant = _quant_slug(meta, model_path)
    if repo_id:
        repo_slug = _slug(repo_id.replace("/", "-"), max_len=72)
        tag = f"seiso-{repo_slug}-{quant}" if quant else f"seiso-{repo_slug}"
    else:
        digest = hashlib.sha256(_normalize_key(model_path).encode("utf-8")).hexdigest()[
            :12
        ]
        tag = f"seiso-{quant}-{digest}" if quant else f"seiso-{digest}"
    return tag[:_MAX_TAG_LEN]


def _coerce_entry(raw: Any) -> dict[str, Any]:
    if isinstance(raw, str):
        return {"tag": raw}
    if isinstance(raw, dict):
        return dict(raw)
    return {}


def load_registry_entries() -> dict[str, dict[str, Any]]:
    path = registry_path()
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for key, raw in data.items():
        entry = _coerce_entry(raw)
        tag = entry.get("tag")
        if tag:
            out[str(key)] = entry
    return out


def load_registry() -> dict[str, str]:
    return {key: str(entry["tag"]) for key, entry in load_registry_entries().items()}


def save_registry(mapping: dict[str, str]) -> None:
    """Backward-compatible tag-only registry writer."""
    save_registry_entries({key: {"tag": tag} for key, tag in mapping.items()})


def save_registry_entries(entries: dict[str, dict[str, Any]]) -> None:
    path = registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        key: {
            **entry,
            "tag": str(entry["tag"]),
        }
        for key, entry in sorted(entries.items())
        if entry.get("tag")
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _persist_entry(
    model_path: str,
    tag: str,
    metadata: dict[str, Any] | None = None,
    *,
    create_skipped: bool = False,
) -> None:
    key = _normalize_key(model_path)
    entries = load_registry_entries()
    meta = _merge_metadata(metadata, model_path=model_path)
    entries[key] = {
        "tag": tag,
        "repo_id": meta.get("repo_id"),
        "gguf_file": meta.get("gguf_file"),
        "format": meta.get("format"),
        "ollama_pull": meta.get("ollama_pull"),
        "create_skipped": create_skipped,
    }
    save_registry_entries(entries)


def lookup_registry_metadata(model_path: str) -> dict[str, Any]:
    key = _normalize_key(model_path)
    entry = load_registry_entries().get(key, {})
    meta: dict[str, Any] = {}
    for field in ("repo_id", "gguf_file", "format", "ollama_pull", "ollama_tag"):
        if entry.get(field):
            meta[field] = entry[field]
    if entry.get("tag"):
        meta.setdefault("ollama_tag", entry["tag"])
    return meta


def resolve_ollama_tag(
    model_path: str,
    metadata: dict[str, Any] | None = None,
    *,
    model_format: str | None = None,
) -> str:
    key = _normalize_key(model_path)
    entries = load_registry_entries()
    if key in entries and entries[key].get("tag"):
        return str(entries[key]["tag"])
    meta = _merge_metadata(metadata, model_path=model_path, model_format=model_format)
    meta = {**lookup_registry_metadata(model_path), **meta}
    repo_id = meta.get("repo_id")
    if isinstance(repo_id, str) and repo_id:
        return default_ollama_tag(key, repo_id=repo_id, metadata=meta)
    return default_ollama_tag(key, metadata=meta)


def resolve_ollama_artifact(
    model_path: str,
    metadata: dict[str, Any] | None = None,
    *,
    model_format: str | None = None,
) -> OllamaArtifact | None:
    """Describe how this inventory path should be served through Ollama."""
    meta = _merge_metadata(metadata, model_path=model_path, model_format=model_format)
    meta = {**lookup_registry_metadata(model_path), **meta}
    tag = resolve_ollama_tag(model_path, meta, model_format=model_format)

    if pull := meta.get("ollama_pull"):
        pull_name = str(pull).strip()
        if pull_name:
            return OllamaArtifact(kind="pull", tag=tag, pull_name=pull_name)

    if meta.get("ollama_tag") and not _path_has_creatable_gguf(model_path, meta, model_format):
        return OllamaArtifact(kind="reference", tag=tag)

    fmt = (model_format or meta.get("format") or "").lower()
    path = Path(model_path).expanduser()
    if fmt == "gguf" or path.suffix.lower() == ".gguf" or _path_has_creatable_gguf(
        model_path, meta, model_format
    ):
        gguf_path = _resolve_gguf_path(model_path, meta)
        if gguf_path is not None:
            return OllamaArtifact(kind="gguf", path=gguf_path, tag=tag)

    if meta.get("ollama_tag") or meta.get("ollama_pull"):
        return OllamaArtifact(kind="reference", tag=tag)

    return None


def _path_has_creatable_gguf(
    model_path: str,
    metadata: dict[str, Any] | None,
    model_format: str | None,
) -> bool:
    return _resolve_gguf_path(model_path, _merge_metadata(metadata, model_path=model_path, model_format=model_format)) is not None


def _resolve_gguf_path(model_path: str, metadata: dict[str, Any]) -> Path | None:
    path = Path(model_path).expanduser()
    if path.is_file() and path.suffix.lower() == ".gguf":
        return path.resolve()
    gguf_file = metadata.get("gguf_file")
    if isinstance(gguf_file, str) and gguf_file.strip():
        candidate = path / gguf_file if path.is_dir() else path.parent / gguf_file
        if candidate.is_file():
            return candidate.resolve()
    if path.is_dir():
        try:
            from seiso.inference.backends import resolve_gguf_file

            return resolve_gguf_file(str(path)).resolve()
        except Exception:
            for candidate in sorted(path.glob("*.gguf")):
                return candidate.resolve()
    return None


def _ollama_available() -> bool:
    from seiso.inference.sidecar_runtime import ollama_registration_available

    return ollama_registration_available()


def _ollama_subprocess_env() -> dict[str, str]:
    from seiso.inference.sidecar_runtime import ollama_cli_host

    env = dict(os.environ)
    env["OLLAMA_HOST"] = ollama_cli_host()
    return env


def _ollama_model_exists(tag: str) -> bool:
    if not shutil.which("ollama"):
        return False
    try:
        proc = subprocess.run(
            ["ollama", "show", tag],
            check=False,
            timeout=30,
            capture_output=True,
            text=True,
            env=_ollama_subprocess_env(),
        )
        return proc.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _run_ollama_create(tag: str, modelfile: Path) -> None:
    cmd = ["ollama", "create", tag, "-f", str(modelfile)]
    logger.info("Registering model with Ollama: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, timeout=900, capture_output=True, text=True, env=_ollama_subprocess_env())


def _run_ollama_pull(pull_name: str) -> None:
    cmd = ["ollama", "pull", pull_name]
    logger.info("Pulling Ollama model: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, timeout=1800, capture_output=True, text=True, env=_ollama_subprocess_env())


def register_model_with_ollama(
    model_path: str,
    *,
    repo_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    model_format: str | None = None,
) -> str:
    """Register or reference a local inventory model in Ollama."""
    meta = _merge_metadata(metadata, model_path=model_path, model_format=model_format)
    if repo_id and not meta.get("repo_id"):
        meta["repo_id"] = repo_id

    artifact = resolve_ollama_artifact(model_path, meta, model_format=model_format)
    if artifact is None:
        raise ValueError(
            f"No Ollama registration strategy for {model_path} "
            f"(format={model_format or meta.get('format')!r})"
        )

    tag = artifact.tag or resolve_ollama_tag(model_path, meta, model_format=model_format)
    _persist_entry(model_path, tag, meta, create_skipped=True)

    if not _ollama_available():
        return tag

    try:
        if artifact.kind == "pull" and artifact.pull_name:
            if not _ollama_model_exists(tag):
                _run_ollama_pull(artifact.pull_name)
            if _ollama_model_exists(tag):
                _persist_entry(model_path, tag, meta, create_skipped=False)
            return tag

        if artifact.kind == "reference":
            if not _ollama_model_exists(tag):
                logger.warning(
                    "Ollama model %s is referenced but not present locally", tag
                )
            _persist_entry(model_path, tag, meta, create_skipped=False)
            return tag

        if artifact.kind == "gguf" and artifact.path is not None:
            if _ollama_model_exists(tag):
                _persist_entry(model_path, tag, meta, create_skipped=False)
                return tag
            modelfile_dir = artifact.path.parent
            write_modelfile(modelfile_dir, artifact.path.name)
            _run_ollama_create(tag, modelfile_dir / "Modelfile")
            _persist_entry(model_path, tag, meta, create_skipped=False)
            return tag
    except Exception as exc:
        logger.warning("Ollama registration failed for %s: %s", model_path, exc)
        _persist_entry(model_path, tag, meta, create_skipped=True)
        return tag

    return tag


def register_gguf_with_ollama(
    model_path: str,
    *,
    repo_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Backward-compatible GGUF registration entry point."""
    return register_model_with_ollama(
        model_path,
        repo_id=repo_id,
        metadata=metadata,
        model_format="gguf",
    )


def ensure_model_registered(
    model_path: str,
    *,
    repo_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    model_format: str | None = None,
) -> str:
    """Return Ollama tag, registering on first use when Ollama is active."""
    meta = _merge_metadata(metadata, model_path=model_path, model_format=model_format)
    meta = {**lookup_registry_metadata(model_path), **meta}
    if repo_id and not meta.get("repo_id"):
        meta["repo_id"] = repo_id

    key = _normalize_key(model_path)
    entries = load_registry_entries()
    if key in entries and entries[key].get("tag"):
        tag = str(entries[key]["tag"])
        if _ollama_available() and not entries[key].get("create_skipped"):
            needs_retry = (
                entries[key].get("format") == "gguf"
                and not _ollama_model_exists(tag)
            )
            if not needs_retry:
                return tag
        if _ollama_available():
            try:
                return register_model_with_ollama(
                    model_path,
                    metadata=meta,
                    model_format=model_format,
                )
            except ValueError:
                return tag
        return tag

    tag = resolve_ollama_tag(model_path, meta, model_format=model_format)
    _persist_entry(model_path, tag, meta, create_skipped=not _ollama_available())

    if not _ollama_available():
        return tag

    try:
        return register_model_with_ollama(
            model_path,
            metadata=meta,
            model_format=model_format,
        )
    except ValueError:
        return tag
    except Exception as exc:
        logger.warning("Ollama ensure failed for %s: %s", model_path, exc)
        return tag


def ensure_gguf_registered(
    model_path: str,
    *,
    repo_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Backward-compatible ensure entry point."""
    return ensure_model_registered(
        model_path,
        repo_id=repo_id,
        metadata=metadata,
        model_format="gguf",
    )


def metadata_for_model_path(
    model_path: str,
    payload_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge registry, payload, and path-derived metadata for Ollama routing."""
    meta = lookup_registry_metadata(model_path)
    if payload_metadata:
        meta = {**meta, **payload_metadata}
    return _merge_metadata(meta, model_path=model_path)
