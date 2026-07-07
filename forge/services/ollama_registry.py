"""Backward-compatible re-export of Ollama registry helpers."""

from seiso.inference.ollama_registry import (
    OllamaArtifact,
    default_ollama_tag,
    ensure_gguf_registered,
    ensure_model_registered,
    load_registry,
    load_registry_entries,
    lookup_registry_metadata,
    metadata_for_model_path,
    register_gguf_with_ollama,
    register_model_with_ollama,
    registry_path,
    resolve_ollama_artifact,
    resolve_ollama_tag,
    save_registry,
    save_registry_entries,
)

__all__ = [
    "OllamaArtifact",
    "default_ollama_tag",
    "ensure_gguf_registered",
    "ensure_model_registered",
    "load_registry",
    "load_registry_entries",
    "lookup_registry_metadata",
    "metadata_for_model_path",
    "register_gguf_with_ollama",
    "register_model_with_ollama",
    "registry_path",
    "resolve_ollama_artifact",
    "resolve_ollama_tag",
    "save_registry",
    "save_registry_entries",
]
