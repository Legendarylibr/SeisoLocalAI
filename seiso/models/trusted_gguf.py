"""Hugging Face GGUF repo helpers — any Hub repo, ranked by popularity."""

from __future__ import annotations

import re
from typing import Any


def is_supported_gguf_repo_candidate(repo_id: str) -> bool:
    repo = repo_id.strip()
    return bool(repo) and "/" in repo


def base_model_from_tags(tags: list[str] | tuple[str, ...]) -> str | None:
    for tag in tags:
        if tag.startswith("base_model:") and not tag.startswith("base_model:quantized:"):
            return tag.split(":", 1)[1]
    return None


def is_trusted_gguf_repo(
    repo_id: str,
    *,
    base_repo_id: str | None = None,
    allow_catalog_mirrors: bool = True,
) -> bool:
    """Return True for any valid Hugging Face model repo id."""
    del base_repo_id, allow_catalog_mirrors
    return is_supported_gguf_repo_candidate(repo_id)


def rank_trusted_gguf_repos(
    repo_ids: list[str],
    *,
    base_repo_id: str | None = None,
    popularity: dict[str, int] | None = None,
) -> list[str]:
    """Sort repo ids by Hub popularity when available."""
    del base_repo_id
    popularity = popularity or {}

    def sort_key(repo_id: str) -> tuple[int, str]:
        return (-int(popularity.get(repo_id, 0)), repo_id.lower())

    supported = [repo_id for repo_id in repo_ids if is_supported_gguf_repo_candidate(repo_id)]
    return sorted(supported, key=sort_key)


def gguf_mirror_candidates(base_repo_id: str) -> list[str]:
    """Naming variants to probe when a base Hub repo does not host GGUF files."""
    repo = base_repo_id.strip()
    if not repo:
        return []
    if "/" not in repo:
        return [repo]

    owner, model_name = repo.split("/", 1)
    title = re.sub(r"(^|[-_/])([a-z])", lambda m: m.group(1) + m.group(2).upper(), model_name)
    mirrors = [
        f"{owner}/{model_name}-GGUF",
        f"{owner}/{title}-GGUF",
        repo,
    ]
    seen: set[str] = set()
    ordered: list[str] = []
    for candidate in mirrors:
        if candidate not in seen:
            seen.add(candidate)
            ordered.append(candidate)
    return ordered


def filter_trusted_gguf_search_results(
    rows: list[dict[str, Any]],
    *,
    base_repo_id: str | None = None,
) -> list[dict[str, Any]]:
    """Keep valid Hub repos and rank them by downloads."""
    del base_repo_id
    supported_rows = [
        row
        for row in rows
        if isinstance(row.get("repo_id"), str)
        and is_supported_gguf_repo_candidate(str(row["repo_id"]))
    ]
    popularity = {
        str(row["repo_id"]): int(row.get("downloads") or 0)
        for row in supported_rows
        if isinstance(row.get("repo_id"), str)
    }
    ordered_ids = rank_trusted_gguf_repos(
        [str(row["repo_id"]) for row in supported_rows],
        popularity=popularity,
    )
    by_id = {str(row["repo_id"]): row for row in supported_rows}
    return [by_id[repo_id] for repo_id in ordered_ids if repo_id in by_id]
