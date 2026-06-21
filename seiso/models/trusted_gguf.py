"""Trusted Hugging Face GGUF publishers and repo trust scoring."""

from __future__ import annotations

from typing import Any

# Well-known quantizers and community mirrors with consistent naming/quality.
TRUSTED_GGUF_PUBLISHERS: frozenset[str] = frozenset(
    {
        "unsloth",
        "bartowski",
        "QuantFactory",
        "lmstudio-community",
        "TheBloke",
        "mradermacher",
        "MaziyarPanahi",
        "mlx-community",
    }
)

# Repo-id substrings that indicate experimental / non-inference artifacts.
_UNSUPPORTED_GGUF_REPO_HINTS: frozenset[str] = frozenset({"dflash", "draft"})

# Explicit GGUF mirrors approved outside the publisher allowlist.
CURATED_GGUF_MIRRORS: frozenset[str] = frozenset(
    {
        "AesSedai/Kimi-K2.7-Code-GGUF",
    }
)

# Higher rank = preferred when multiple trusted mirrors exist.
_PUBLISHER_RANK: dict[str, int] = {
    "unsloth": 100,
    "bartowski": 90,
    "QuantFactory": 85,
    "lmstudio-community": 80,
    "mlx-community": 75,
    "TheBloke": 70,
    "mradermacher": 65,
    "MaziyarPanahi": 60,
}


def gguf_repo_owner(repo_id: str) -> str:
    return repo_id.split("/", 1)[0]


def is_supported_gguf_repo_candidate(repo_id: str) -> bool:
    lowered = repo_id.lower()
    return not any(hint in lowered for hint in _UNSUPPORTED_GGUF_REPO_HINTS)


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
    """Return True when a GGUF repo is from a curated or reputable source."""
    if not is_supported_gguf_repo_candidate(repo_id):
        return False

    owner = gguf_repo_owner(repo_id)
    if owner in TRUSTED_GGUF_PUBLISHERS:
        return True

    if base_repo_id:
        base_owner = gguf_repo_owner(base_repo_id)
        if owner.lower() == base_owner.lower():
            return True

    if allow_catalog_mirrors and repo_id in CURATED_GGUF_MIRRORS:
        return True

    return False


def gguf_repo_trust_rank(repo_id: str, *, base_repo_id: str | None = None) -> int:
    """Higher = more trusted/preferred. Returns -1 for untrusted repos."""
    if not is_trusted_gguf_repo(repo_id, base_repo_id=base_repo_id):
        return -1

    owner = gguf_repo_owner(repo_id)
    rank = _PUBLISHER_RANK.get(owner, 50)
    if base_repo_id and owner.lower() == gguf_repo_owner(base_repo_id).lower():
        rank += 10
    return rank


def rank_trusted_gguf_repos(
    repo_ids: list[str],
    *,
    base_repo_id: str | None = None,
    popularity: dict[str, int] | None = None,
) -> list[str]:
    """Sort repo ids by trust rank, then Hub popularity when available."""
    popularity = popularity or {}

    def sort_key(repo_id: str) -> tuple[int, int, str]:
        return (
            -gguf_repo_trust_rank(repo_id, base_repo_id=base_repo_id),
            -int(popularity.get(repo_id, 0)),
            repo_id.lower(),
        )

    trusted = [repo_id for repo_id in repo_ids if gguf_repo_trust_rank(repo_id, base_repo_id=base_repo_id) >= 0]
    return sorted(trusted, key=sort_key)


def gguf_mirror_candidates(base_repo_id: str) -> list[str]:
    """Ordered mirror repo ids to probe for a base Hugging Face model."""
    import re

    model_name = base_repo_id.split("/")[-1]
    title = re.sub(r"(^|[-_/])([a-z])", lambda m: m.group(1) + m.group(2).upper(), model_name)
    mirrors = [
        f"unsloth/{model_name}-GGUF",
        f"unsloth/{title}-GGUF",
        f"bartowski/{model_name}-GGUF",
        f"bartowski/{title}-GGUF",
        f"QuantFactory/{model_name}-GGUF",
        f"QuantFactory/{title}-GGUF",
        f"lmstudio-community/{model_name}-GGUF",
        f"lmstudio-community/{title}-GGUF",
    ]
    if "Qwen" in base_repo_id:
        mirrors.insert(4, f"bartowski/Qwen_{model_name}-GGUF")
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
    """Keep only reputable GGUF repos and rank them for display/download."""
    trusted_rows = [
        row
        for row in rows
        if isinstance(row.get("repo_id"), str)
        and is_trusted_gguf_repo(str(row["repo_id"]), base_repo_id=base_repo_id)
    ]
    popularity = {
        str(row["repo_id"]): int(row.get("downloads") or 0)
        for row in trusted_rows
        if isinstance(row.get("repo_id"), str)
    }
    ordered_ids = rank_trusted_gguf_repos(
        [str(row["repo_id"]) for row in trusted_rows],
        base_repo_id=base_repo_id,
        popularity=popularity,
    )
    by_id = {str(row["repo_id"]): row for row in trusted_rows}
    return [by_id[repo_id] for repo_id in ordered_ids if repo_id in by_id]
