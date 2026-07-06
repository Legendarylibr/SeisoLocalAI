"""Resolve trainable Hub mirrors when official repos are gated."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from seiso.models.hub_errors import format_hub_error, is_gated_hub_error

logger = logging.getLogger(__name__)

# Open training mirror providers — same model slug, different org (no per-model map).
_TRAINABLE_MIRROR_PROVIDERS: tuple[str, ...] = ("unsloth",)


def _trainable_mirror_candidates(repo_id: str) -> list[str]:
    repo = repo_id.strip()
    if "/" not in repo:
        return []
    _owner, name = repo.split("/", 1)
    seen: set[str] = {repo.lower()}
    candidates: list[str] = []
    for provider in _TRAINABLE_MIRROR_PROVIDERS:
        candidate = f"{provider}/{name}"
        key = candidate.lower()
        if key not in seen:
            seen.add(key)
            candidates.append(candidate)
    return candidates


def _probe_hub_config_download(
    repo_id: str,
    *,
    token: str | None,
) -> Literal["ok", "gated"]:
    if Path(repo_id).exists():
        return "ok"
    try:
        from huggingface_hub import hf_hub_download

        hf_hub_download(repo_id, "config.json", token=token)
        return "ok"
    except Exception as exc:
        if is_gated_hub_error(exc):
            return "gated"
        raise ValueError(
            format_hub_error(exc, context="download", repo_id=repo_id)
        ) from exc


def resolve_trainable_hub_id(
    model_id: str,
    *,
    token: str | None = None,
) -> tuple[str, str | None]:
    """
    Return a Hub repo id that can actually download weights for training.

    When the requested repo is gated (license not accepted), probe known open
    mirror providers with the same model slug. Transient Hub/network errors are
    raised instead of silently switching mirrors.
    """
    repo = model_id.strip()
    if not repo or Path(repo).exists():
        return repo, None

    if _probe_hub_config_download(repo, token=token) == "ok":
        return repo, None

    for mirror in _trainable_mirror_candidates(repo):
        if _probe_hub_config_download(mirror, token=token) == "ok":
            note = (
                f"{repo} is gated on Hugging Face (accept the license at "
                f"https://huggingface.co/{repo} to use the official weights). "
                f"Training with mirror {mirror} instead."
            )
            logger.warning(note)
            return mirror, note

    return repo, (
        f"Cannot download {repo}. If it is gated, open "
        f"https://huggingface.co/{repo} and accept the license, then retry."
    )
