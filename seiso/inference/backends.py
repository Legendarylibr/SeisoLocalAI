"""Inference backend selection for local models."""

from __future__ import annotations

from pathlib import Path

from seiso.models.loader import Backend, detect_backend

BackendName = str

BACKEND_LLAMACPP = "llamacpp"
BACKEND_OLLAMA = "ollama"
BACKEND_MLX = "mlx"
BACKEND_TORCH = "torch"
BACKEND_AUTO = "auto"


def _is_gguf_path(model_path: str) -> bool:
    path = Path(model_path)
    if path.suffix.lower() == ".gguf":
        return True
    return path.is_dir() and any(path.glob("*.gguf"))


def resolve_gguf_file(model_path: str) -> Path:
    """Pick a single GGUF file from a path or directory."""
    path = Path(model_path).expanduser()
    if path.is_file() and path.suffix.lower() == ".gguf":
        return path.resolve()
    if path.is_dir():
        candidates = sorted(path.glob("*.gguf"), key=lambda p: p.stat().st_size, reverse=True)
        if candidates:
            return candidates[0].resolve()
    raise ValueError(f"No GGUF file found at {model_path}")


def recommend_backend(*, model_path: str, model_format: str | None = None) -> BackendName:
    """Pick the default local inference engine from model path/format."""
    fmt = (model_format or "").lower()
    path = Path(model_path)
    if fmt == "gguf" or _is_gguf_path(model_path):
        return BACKEND_LLAMACPP
    if fmt in {"safetensors", "bin"} or path.is_dir():
        backend = detect_backend()
        if backend == Backend.MLX:
            return BACKEND_MLX
        return BACKEND_TORCH
    if path.suffix.lower() == ".gguf":
        return BACKEND_LLAMACPP
    return BACKEND_TORCH


def match_ollama_name(
    *,
    model_path: str,
    model_name: str,
    ollama_names: set[str],
) -> str | None:
    """Find an Ollama tag that likely corresponds to a local GGUF/checkpoint."""
    stems = {Path(model_path).stem.lower(), model_name.lower()}
    for tag in ollama_names:
        base = tag.split(":")[0].lower()
        for stem in stems:
            if base == stem or base.startswith(stem) or stem in base:
                return tag
    return None


def available_backends(*, model_path: str, model_format: str | None, ollama_names: set[str]) -> list[BackendName]:
    """Backends that can serve this inventory model."""
    primary = recommend_backend(model_path=model_path, model_format=model_format)
    options = [primary]
    if primary == BACKEND_LLAMACPP and match_ollama_name(
        model_path=model_path,
        model_name=Path(model_path).stem,
        ollama_names=ollama_names,
    ):
        options.append(BACKEND_OLLAMA)
    return options


def resolve_local_backend(
    *,
    model_path: str,
    model_format: str | None,
    requested: str | None,
) -> BackendName:
    """Resolve auto/requested backend for a filesystem model."""
    choice = (requested or BACKEND_AUTO).lower()
    if choice == BACKEND_AUTO:
        return recommend_backend(model_path=model_path, model_format=model_format)
    if choice in {BACKEND_LLAMACPP, BACKEND_OLLAMA, BACKEND_MLX, BACKEND_TORCH}:
        return choice
    raise ValueError(f"Unsupported inference backend: {requested}")


def prepare_model_path(model_path: str, backend: BackendName) -> str:
    """Normalize model path (e.g. pick a GGUF file inside a directory)."""
    if backend == BACKEND_LLAMACPP:
        return str(resolve_gguf_file(model_path))
    return model_path
