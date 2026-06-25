"""Detect Hugging Face snapshots suitable for LoRA/SFT training."""

from __future__ import annotations

import re
from pathlib import Path

from seiso.models.trusted_gguf import is_trusted_gguf_repo

_TRAINABLE_WEIGHT_SUFFIXES = frozenset({".safetensors", ".bin"})
_GGUF_REPO_RE = re.compile(r"(?:^|/)[^/]*-gguf(?:$|/|-)", re.I)

GGUF_ONLY_REPO_MESSAGE = (
    "This model cache is GGUF-only (chat/inference weights). LoRA/QLoRA training needs a "
    "safetensors or PyTorch checkpoint — pick a non-GGUF Hugging Face repo or download "
    "the safetensors variant from the catalog."
)


def is_gguf_only_repo_id(repo_id: str, tags: list[str] | tuple[str, ...] | None = None) -> bool:
    """True when *repo_id* points at a GGUF inference mirror, not a trainable checkpoint."""
    lowered = repo_id.strip().lower()
    if not lowered:
        return False
    if _GGUF_REPO_RE.search(lowered) or lowered.endswith("-gguf"):
        return True
    if is_trusted_gguf_repo(repo_id):
        return True
    tag_set = {t.lower() for t in (tags or ())}
    return bool("gguf" in tag_set and "safetensors" not in tag_set and "pytorch" not in tag_set)


def trainable_weight_files(root: Path) -> list[Path]:
    """Return weight files under *root* that transformers can load for training."""
    path = root.expanduser()
    if path.is_file():
        return [path] if path.suffix.lower() in _TRAINABLE_WEIGHT_SUFFIXES else []

    files: list[Path] = []
    for pattern in ("*.safetensors", "*.bin"):
        files.extend(p for p in path.rglob(pattern) if p.is_file())
    return files


def snapshot_has_trainable_weights(root: Path) -> bool:
    """True when *root* contains at least one safetensors/bin weight file."""
    return bool(trainable_weight_files(root))
