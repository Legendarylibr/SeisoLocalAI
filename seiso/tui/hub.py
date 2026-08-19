"""Hugging Face Hub + local inventory for the Seiso TUI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from seiso.tui.offline import (
    discover_local_gguf,
    format_size,
    local_model_label,
    looks_like_digest_name,
)


@dataclass(frozen=True, slots=True)
class HubRow:
    key: str
    source: str  # local | hub
    title: str
    repo_id: str
    path: Path | None
    size_label: str
    downloads: int | None
    likes: int | None
    family: str
    task: str
    status: str
    subtitle: str


def _commify(n: int | None) -> str:
    if n is None:
        return "—"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


def local_rows(data_dir: Path) -> list[HubRow]:
    rows: list[HubRow] = []
    for item in discover_local_gguf(data_dir):
        rows.append(
            HubRow(
                key=str(item.path),
                source="local",
                title=item.label,
                repo_id=item.label,
                path=item.path,
                size_label=format_size(item.size_bytes),
                downloads=None,
                likes=None,
                family="gguf",
                task="chat",
                status="ready",
                subtitle=str(item.path),
            )
        )
    return rows


def _catalog_row(entry: dict[str, Any], *, local_names: set[str]) -> HubRow:
    repo = str(entry.get("repo_id") or entry.get("name") or "")
    name = str(entry.get("name") or repo.split("/")[-1] or repo)
    ready = name.lower() in local_names or repo.lower() in local_names
    downloads = entry.get("downloads") if isinstance(entry.get("downloads"), int) else None
    raw_tags = entry.get("tags")
    tags: list[str] = [str(t) for t in raw_tags[:3]] if isinstance(raw_tags, list) else []
    tag_s = " · ".join(tags)
    gguf = entry.get("gguf_repo")
    subtitle_bits = [repo]
    if gguf and gguf != repo:
        subtitle_bits.append(f"GGUF {gguf}")
    if tag_s:
        subtitle_bits.append(tag_s)
    return HubRow(
        key=f"hub:{repo}",
        source="hub",
        title=name,
        repo_id=repo,
        path=None,
        size_label=str(entry.get("params") or entry.get("quant") or "Hub"),
        downloads=downloads,
        likes=None,
        family=str(entry.get("family") or ""),
        task=str(entry.get("task") or ""),
        status="ready" if ready else "remote",
        subtitle=" · ".join(subtitle_bits),
    )


def _gguf_search_row(entry: dict[str, Any], *, local_names: set[str]) -> HubRow:
    repo = str(entry.get("repo_id") or "")
    name = repo.split("/")[-1] if repo else "model"
    ready = name.lower() in local_names or repo.lower() in local_names
    downloads = entry.get("downloads") if isinstance(entry.get("downloads"), int) else None
    likes = entry.get("likes") if isinstance(entry.get("likes"), int) else None
    return HubRow(
        key=f"gguf:{repo}",
        source="hub",
        title=name,
        repo_id=repo,
        path=None,
        size_label="GGUF",
        downloads=downloads,
        likes=likes,
        family="gguf",
        task="chat",
        status="ready" if ready else "remote",
        subtitle=repo,
    )


def search_hub(
    query: str,
    *,
    data_dir: Path,
    limit: int = 16,
    catalog_search: Any = None,
    gguf_search: Any = None,
) -> tuple[list[HubRow], list[HubRow], str | None]:
    """Return (local, remote, error). Remote is live Hugging Face, not only downloads."""
    local = local_rows(data_dir)
    local_names = {row.title.lower() for row in local} | {row.repo_id.lower() for row in local}
    error: str | None = None
    remote: list[HubRow] = []
    seen: set[str] = set()

    if catalog_search is None:
        from seiso.models.catalog import search_catalog

        catalog_search = search_catalog
    if gguf_search is None:
        from forge.services.hf_hub_search import search_huggingface_gguf_repos

        gguf_search = search_huggingface_gguf_repos

    q = query.strip()
    try:
        catalog = catalog_search(q, limit=limit)
        models = getattr(catalog, "models", catalog) if catalog is not None else []
        if isinstance(models, list):
            for entry in models:
                if not isinstance(entry, dict):
                    continue
                row = _catalog_row(entry, local_names=local_names)
                if not row.repo_id or row.repo_id in seen:
                    continue
                seen.add(row.repo_id)
                remote.append(row)
    except Exception as exc:
        error = f"Catalog search failed: {exc}"

    gguf_q = q or "instruct"
    try:
        gguf_hits = gguf_search(query=gguf_q, limit=limit, trusted_only=False)
        if isinstance(gguf_hits, list):
            for entry in gguf_hits:
                if not isinstance(entry, dict):
                    continue
                row = _gguf_search_row(entry, local_names=local_names)
                if not row.repo_id or row.repo_id in seen:
                    continue
                seen.add(row.repo_id)
                remote.append(row)
    except Exception as exc:
        extra = f"GGUF search failed: {exc}"
        error = f"{error} · {extra}" if error else extra

    remote.sort(
        key=lambda row: (
            0 if row.status == "ready" else 1,
            -(row.downloads or 0),
            row.title.lower(),
        )
    )
    return local, remote[: max(1, limit)], error


def download_hub_repo(repo_id: str, *, data_dir: Path) -> HubRow:
    from forge.services.hf_hub import download_gguf

    cache_dir = data_dir / "hf_cache"
    result = download_gguf(repo_id, cache_dir=cache_dir, token=None)
    path = Path(str(result.get("path") or ""))
    if path.is_dir():
        ggufs = sorted(path.glob("*.gguf"))
        path = ggufs[0] if ggufs else path
    size = format_size(path.stat().st_size) if path.is_file() else "downloaded"
    title = local_model_label(path) if path.name else repo_id
    if looks_like_digest_name(title):
        title = repo_id
    return HubRow(
        key=str(path),
        source="local",
        title=title or repo_id,
        repo_id=repo_id,
        path=path if path.exists() else None,
        size_label=size,
        downloads=None,
        likes=None,
        family="gguf",
        task="chat",
        status="ready",
        subtitle=str(path),
    )


def combined_rows(local: list[HubRow], remote: list[HubRow]) -> list[HubRow]:
    return [*local, *remote]


def format_downloads(n: int | None) -> str:
    return _commify(n)
