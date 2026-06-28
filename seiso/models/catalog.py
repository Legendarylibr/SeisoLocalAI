"""Hugging Face Hub model search — live queries via huggingface_hub."""

from __future__ import annotations

import math
import re
import urllib.parse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone

from seiso.compat import StrEnum
from seiso.models.hub_errors import format_hub_error
from seiso.models.trainable_snapshot import is_gguf_only_repo_id

_DEFAULT_LIMIT = 50
_MAX_LIMIT = 100
_HUB_API_MODELS = "https://huggingface.co/api/models"

_SKIP_PIPELINE_TAGS = frozenset(
    {
        "text-to-speech",
        "text-to-audio",
        "automatic-speech-recognition",
        "image-to-video",
        "video-classification",
        "object-detection",
        "depth-estimation",
    }
)


class HubSearchError(Exception):
    """Hub model search failed (network, rate limit, auth, etc.)."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class CatalogSearchResult:
    models: list[dict]
    next_cursor: str | None = None


class ModelFamily(StrEnum):
    LLAMA = "llama"
    QWEN = "qwen"
    GEMMA = "gemma"
    PHI = "phi"
    MISTRAL = "mistral"
    DEEPSEEK = "deepseek"
    KIMI = "kimi"
    GLM = "glm"
    OTHER = "other"


class ModelTask(StrEnum):
    CHAT = "chat"
    CODE = "code"
    VISION = "vision"
    EMBEDDING = "embedding"
    BASE = "base"


_FAMILY_NEEDLES: tuple[tuple[ModelFamily, tuple[str, ...]], ...] = (
    (ModelFamily.QWEN, ("qwen",)),
    (ModelFamily.LLAMA, ("llama", "meta-llama")),
    (ModelFamily.GEMMA, ("gemma", "google/gemma")),
    (ModelFamily.PHI, ("phi", "microsoft/phi")),
    (ModelFamily.MISTRAL, ("mistral", "devstral", "mixtral")),
    (ModelFamily.DEEPSEEK, ("deepseek",)),
    (ModelFamily.KIMI, ("kimi", "moonshot")),
    (ModelFamily.GLM, ("glm", "zai-org")),
)


@dataclass(frozen=True)
class CatalogEntry:
    repo_id: str
    name: str
    family: ModelFamily
    params: str
    task: ModelTask
    quant: str
    tags: tuple[str, ...] = ()
    gguf_repo: str | None = None
    priority: int = 50
    downloads: int | None = None


def _format_hub_search_error(exc: Exception, *, status_code: int | None = None) -> str:
    return format_hub_error(exc, context="search", status_code=status_code)


def _parse_next_link(link_header: str | None) -> str | None:
    if not link_header:
        return None
    for segment in link_header.split(","):
        segment = segment.strip()
        if 'rel="next"' in segment:
            match = re.search(r"<([^>]+)>", segment)
            if match:
                url = match.group(1)
                return url if url.startswith("https://") else None
    return None


def _cursor_from_link(link_header: str | None) -> str | None:
    next_url = _parse_next_link(link_header)
    if not next_url:
        return None
    parsed = urllib.parse.urlparse(next_url)
    values = urllib.parse.parse_qs(parsed.query).get("cursor")
    return values[0] if values else None


def _fetch_hub_page(
    *,
    filter_tag: str | None = None,
    pipeline_tag: str | None = None,
    search: str | None = None,
    sort: str = "downloads",
    limit: int = _DEFAULT_LIMIT,
    cursor: str | None = None,
    token: str | None = None,
) -> tuple[list[dict], str | None]:
    """Fetch one Hub models page using huggingface_hub (returns rows + next cursor)."""
    from huggingface_hub.errors import HfHubHTTPError
    from huggingface_hub.utils import build_hf_headers, get_session, hf_raise_for_status

    params: dict[str, str | int] = {
        "sort": sort,
        "limit": max(1, min(limit, _MAX_LIMIT)),
        "full": "false",
    }
    if filter_tag:
        params["filter"] = filter_tag
    if pipeline_tag:
        params["pipeline_tag"] = pipeline_tag
    if search:
        params["search"] = search.strip()
    if cursor:
        params["cursor"] = cursor

    headers = build_hf_headers(token=token)
    session = get_session()
    try:
        response = session.get(_HUB_API_MODELS, params=params, headers=headers)
        hf_raise_for_status(response)
        payload = response.json()
        next_cursor = _cursor_from_link(response.headers.get("link"))
    except HfHubHTTPError as exc:
        status = exc.response.status_code if exc.response is not None else None
        raise HubSearchError(
            _format_hub_search_error(exc, status_code=status), status_code=status
        ) from exc
    except Exception as exc:
        raise HubSearchError(_format_hub_search_error(exc)) from exc

    if not isinstance(payload, list):
        return [], None
    rows = [row for row in payload if isinstance(row, dict)]
    return rows, next_cursor


def _hub_search_text(query: str, family: str | None) -> str | None:
    parts: list[str] = []
    if query.strip():
        parts.append(query.strip())
    if family:
        parts.append(family)
    return " ".join(parts) if parts else None


def _query_hub_page(
    *,
    query: str = "",
    family: str | None = None,
    task: str | None = None,
    limit: int,
    cursor: str | None = None,
    token: str | None = None,
) -> tuple[list[dict], str | None]:
    hf_search = _hub_search_text(query, family)
    if task == ModelTask.EMBEDDING.value:
        return _fetch_hub_page(
            pipeline_tag="feature-extraction",
            search=hf_search,
            limit=limit,
            cursor=cursor,
            token=token,
        )
    return _fetch_hub_page(
        pipeline_tag="text-generation",
        search=hf_search,
        limit=limit,
        cursor=cursor,
        token=token,
    )


def _query_trainable_hub_page(
    *,
    query: str = "",
    family: str | None = None,
    task: str | None = None,
    limit: int,
    cursor: str | None = None,
    token: str | None = None,
) -> tuple[list[dict], str | None]:
    hf_search = _hub_search_text(query, family)
    if task == ModelTask.EMBEDDING.value:
        return _fetch_hub_page(
            pipeline_tag="feature-extraction",
            search=hf_search,
            limit=limit,
            cursor=cursor,
            token=token,
        )
    return _fetch_hub_page(
        pipeline_tag="text-generation",
        search=hf_search,
        limit=limit,
        cursor=cursor,
        token=token,
    )


def _model_info_to_row(info: object) -> dict:
    created = getattr(info, "created_at", None) or getattr(info, "last_modified", None)
    created_at = created.isoformat() if hasattr(created, "isoformat") else created
    tags = getattr(info, "tags", None) or []
    return {
        "id": getattr(info, "id", None),
        "modelId": getattr(info, "id", None),
        "downloads": getattr(info, "downloads", None),
        "pipeline_tag": getattr(info, "pipeline_tag", None),
        "tags": list(tags) if isinstance(tags, list) else [],
        "createdAt": created_at,
    }


def _parse_iso_ts(value: str | None) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    try:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        return datetime.fromisoformat(value).astimezone(timezone.utc)
    except ValueError:
        return None


def _infer_family(repo_id: str, tags: list[str]) -> ModelFamily:
    hay = f"{repo_id} {' '.join(tags)}".lower()
    for family, needles in _FAMILY_NEEDLES:
        if any(needle in hay for needle in needles):
            return family
    return ModelFamily.OTHER


def _infer_task(repo_id: str, pipeline_tag: str | None, tags: list[str]) -> ModelTask:
    tag_set = {t.lower() for t in tags}
    hay = f"{repo_id} {' '.join(tags)}".lower()
    if pipeline_tag == "feature-extraction" or "embedding" in tag_set:
        return ModelTask.EMBEDDING
    if (
        pipeline_tag == "image-text-to-text"
        or "vision" in tag_set
        or "multimodal" in tag_set
    ):
        return ModelTask.VISION
    if any(k in hay for k in ("coder", "code-", "-code", "coding", "devstral")):
        return ModelTask.CODE
    return ModelTask.CHAT


def _infer_params(repo_id: str, tags: list[str]) -> str:
    for source in (repo_id, " ".join(tags)):
        match = re.search(r"(\d+(?:\.\d+)?)\s*[bB]\b", source)
        if match:
            return f"{match.group(1)}B"
        match = re.search(r"(\d+(?:\.\d+)?)\s*[mM]\b", source)
        if match:
            return f"{match.group(1)}M"
    return "?"


def _display_name(repo_id: str, tags: list[str]) -> str:
    for tag in tags:
        if tag.startswith("base_model:") and not tag.startswith(
            "base_model:quantized:"
        ):
            base = tag.split(":", 1)[1]
            return base.split("/")[-1].replace("-", " ").replace("_", " ")
    slug = repo_id.split("/")[-1]
    slug = re.sub(r"(?i)-gguf$", "", slug)
    slug = re.sub(r"[_-]+", " ", slug).strip()
    return slug or repo_id


def _is_supported_repo(repo_id: str) -> bool:
    repo = repo_id.strip()
    return bool(repo) and "/" in repo


def is_gguf_hub_repo(
    repo_id: str, tags: list[str] | tuple[str, ...] | None = None
) -> bool:
    """True when a Hub row is a GGUF artifact (tags or naming)."""
    return _is_gguf_hub_repo(repo_id, list(tags or ()))


def _is_gguf_hub_repo(repo_id: str, tags: list[str]) -> bool:
    """True when a Hub row is a GGUF model (HF gguf filter or naming/tags)."""
    tag_set = {t.lower() for t in tags}
    if "gguf" in tag_set:
        return True
    lowered = repo_id.lower()
    return "-gguf" in lowered or lowered.endswith("gguf")


def _compute_priority(downloads: int, created_at: str | None, tags: list[str]) -> int:
    dl_score = min(88, int(math.log10(max(downloads, 1)) * 14))
    created = _parse_iso_ts(created_at)
    recency = 0
    if created:
        age_days = (datetime.now(timezone.utc) - created).days
        if age_days <= 30:
            recency = 12
        elif age_days <= 90:
            recency = 8
        elif age_days <= 180:
            recency = 4
    tag_bonus = 5 if "popular" in tags else 0
    return min(100, dl_score + recency + tag_bonus)


def _hub_row_to_entry(
    row: dict, *, force_task: ModelTask | None = None
) -> CatalogEntry | None:
    repo_id = row.get("id") or row.get("modelId")
    if not isinstance(repo_id, str) or not repo_id.strip():
        return None
    if not _is_supported_repo(repo_id):
        return None

    tags_raw = row.get("tags")
    tags = (
        [t for t in tags_raw if isinstance(t, str)]
        if isinstance(tags_raw, list)
        else []
    )
    pipeline_tag = (
        row.get("pipeline_tag") if isinstance(row.get("pipeline_tag"), str) else None
    )
    if pipeline_tag in _SKIP_PIPELINE_TAGS:
        return None

    task = force_task or _infer_task(repo_id, pipeline_tag, tags)
    is_gguf = _is_gguf_hub_repo(repo_id, tags)
    downloads = row.get("downloads") if isinstance(row.get("downloads"), int) else 0
    created_at = row.get("createdAt") if isinstance(row.get("createdAt"), str) else None
    family = _infer_family(repo_id, tags)
    params = _infer_params(repo_id, tags)
    name = _display_name(repo_id, tags)
    catalog_tags = tuple(dict.fromkeys(["popular", *tags[:6]]))
    priority = _compute_priority(downloads, created_at, list(catalog_tags))

    return CatalogEntry(
        repo_id=repo_id,
        name=name,
        family=family,
        params=params,
        task=task,
        quant=(
            "Q4_K_M" if is_gguf else ("F16" if task == ModelTask.EMBEDDING else "bf16")
        ),
        tags=catalog_tags,
        gguf_repo=repo_id if is_gguf else None,
        priority=priority,
        downloads=downloads,
    )


def _hub_row_to_trainable_entry(
    row: dict, *, force_task: ModelTask | None = None
) -> CatalogEntry | None:
    repo_id = row.get("id") or row.get("modelId")
    if not isinstance(repo_id, str) or not repo_id.strip():
        return None
    if not _is_supported_repo(repo_id):
        return None

    tags_raw = row.get("tags")
    tags = (
        [t for t in tags_raw if isinstance(t, str)]
        if isinstance(tags_raw, list)
        else []
    )
    if is_gguf_only_repo_id(repo_id, tags):
        return None

    pipeline_tag = (
        row.get("pipeline_tag") if isinstance(row.get("pipeline_tag"), str) else None
    )
    if pipeline_tag in _SKIP_PIPELINE_TAGS:
        return None

    task = force_task or _infer_task(repo_id, pipeline_tag, tags)
    if task == ModelTask.EMBEDDING and force_task != ModelTask.EMBEDDING:
        return None
    if task == ModelTask.VISION:
        return None

    downloads = row.get("downloads") if isinstance(row.get("downloads"), int) else 0
    created_at = row.get("createdAt") if isinstance(row.get("createdAt"), str) else None
    family = _infer_family(repo_id, tags)
    params = _infer_params(repo_id, tags)
    name = _display_name(repo_id, tags)
    catalog_tags = tuple(dict.fromkeys(["trainable", *tags[:6]]))
    priority = _compute_priority(downloads, created_at, list(catalog_tags))
    hay = f"{repo_id} {' '.join(tags)}".lower()
    if "instruct" in hay or "-it" in hay or "chat" in hay:
        priority = min(100, priority + 8)

    return CatalogEntry(
        repo_id=repo_id,
        name=name,
        family=family,
        params=params,
        task=task,
        quant="bf16",
        tags=catalog_tags,
        gguf_repo=None,
        priority=priority,
        downloads=downloads,
    )


def _parse_param_size(params: str) -> float:
    p = params.strip().upper()
    if p.endswith("T"):
        return float(p[:-1]) * 1000
    if p.endswith("B"):
        return float(p[:-1])
    if p.endswith("M"):
        return float(p[:-1]) / 1000
    if p == "?":
        return float("inf")
    return float(p)


def _entry_to_dict(e: CatalogEntry) -> dict:
    row = {
        "repo_id": e.repo_id,
        "name": e.name,
        "family": e.family.value,
        "params": e.params,
        "task": e.task.value,
        "quant": e.quant,
        "tags": list(e.tags),
        "gguf_repo": e.gguf_repo,
        "featured": e.priority >= 80,
        "priority": e.priority,
    }
    if e.downloads is not None:
        row["downloads"] = e.downloads
    return row


def _matches_task(entry: CatalogEntry, task: str) -> bool:
    task_l = task.lower()
    return entry.task.value == task_l or (
        task_l == "chat" and entry.task == ModelTask.CODE and "popular" in entry.tags
    )


def _boost_score(entry: CatalogEntry, query: str) -> float:
    """Soft ranking boost — never drops HF search results."""
    base = float(entry.priority)
    if not query.strip():
        return base
    q = query.lower().strip()
    name_l = entry.name.lower()
    repo_l = entry.repo_id.lower()
    haystack = f"{name_l} {repo_l} {' '.join(entry.tags)}"
    score = base
    if q in repo_l:
        score += 40
    elif q in name_l:
        score += 25
    elif q in haystack:
        score += 12
    for tok in q.split():
        if tok in repo_l:
            score += 20
        elif tok in haystack:
            score += 8
    return score


def search_catalog(
    query: str = "",
    family: str | None = None,
    task: str | None = None,
    max_params: str | None = None,
    *,
    limit: int = _DEFAULT_LIMIT,
    cursor: str | None = None,
    token: str | None = None,
) -> CatalogSearchResult:
    """Query Hugging Face Hub live via huggingface_hub; popular models rank first."""
    limit = max(1, min(limit, _MAX_LIMIT))
    rows, next_cursor = _query_hub_page(
        query=query,
        family=family,
        task=task,
        limit=limit,
        cursor=cursor,
        token=token,
    )

    force_task = ModelTask.EMBEDDING if task == ModelTask.EMBEDDING.value else None
    entries: list[CatalogEntry] = []
    seen: set[str] = set()
    for row in rows:
        entry = _hub_row_to_entry(row, force_task=force_task)
        if entry is None or entry.repo_id in seen:
            continue
        seen.add(entry.repo_id)
        entries.append(entry)

    scored: list[tuple[float, dict]] = []
    for entry in entries:
        if family and not query.strip() and entry.family.value != family.lower():
            continue
        if task and not _matches_task(entry, task):
            continue
        if max_params and _parse_param_size(entry.params) > _parse_param_size(
            max_params
        ):
            continue
        scored.append((_boost_score(entry, query), _entry_to_dict(entry)))

    if query.strip():
        scored.sort(
            key=lambda pair: (
                -pair[0],
                -(pair[1].get("downloads") or 0),
                pair[1]["name"],
            )
        )
    else:
        scored.sort(
            key=lambda pair: (
                -(pair[1].get("downloads") or 0),
                -pair[0],
                pair[1]["name"],
            )
        )

    return CatalogSearchResult(
        models=[item for _, item in scored],
        next_cursor=next_cursor,
    )


def search_trainable_catalog(
    query: str = "",
    family: str | None = None,
    task: str | None = None,
    max_params: str | None = None,
    *,
    limit: int = _DEFAULT_LIMIT,
    cursor: str | None = None,
    token: str | None = None,
) -> CatalogSearchResult:
    """Query Hugging Face Hub for safetensors checkpoints suitable for LoRA/SFT."""
    limit = max(1, min(limit, _MAX_LIMIT))
    rows, next_cursor = _query_trainable_hub_page(
        query=query,
        family=family,
        task=task,
        limit=limit,
        cursor=cursor,
        token=token,
    )

    force_task = ModelTask.EMBEDDING if task == ModelTask.EMBEDDING.value else None
    entries: list[CatalogEntry] = []
    seen: set[str] = set()
    for row in rows:
        entry = _hub_row_to_trainable_entry(row, force_task=force_task)
        if entry is None or entry.repo_id in seen:
            continue
        seen.add(entry.repo_id)
        entries.append(entry)

    scored: list[tuple[float, dict]] = []
    for entry in entries:
        if family and not query.strip() and entry.family.value != family.lower():
            continue
        if task and not _matches_task(entry, task):
            continue
        if max_params and _parse_param_size(entry.params) > _parse_param_size(
            max_params
        ):
            continue
        scored.append((_boost_score(entry, query), _entry_to_dict(entry)))

    if query.strip():
        scored.sort(
            key=lambda pair: (
                -pair[0],
                -(pair[1].get("downloads") or 0),
                pair[1]["name"],
            )
        )
    else:
        scored.sort(
            key=lambda pair: (
                -(pair[1].get("downloads") or 0),
                -pair[0],
                pair[1]["name"],
            )
        )

    return CatalogSearchResult(
        models=[item for _, item in scored],
        next_cursor=next_cursor,
    )


def diversify_by_family(models: list[dict]) -> list[dict]:
    if len(models) <= 1:
        return list(models)

    by_family: dict[str, list[dict]] = defaultdict(list)
    for m in models:
        by_family[m.get("family") or "other"].append(m)

    family_order = sorted(
        by_family.keys(),
        key=lambda fam: (
            -(by_family[fam][0].get("priority") or 0) if by_family[fam] else 0
        ),
    )
    indices = {fam: 0 for fam in family_order}
    diversified: list[dict] = []

    while True:
        added = False
        for fam in family_order:
            idx = indices[fam]
            if idx < len(by_family[fam]):
                diversified.append(by_family[fam][idx])
                indices[fam] = idx + 1
                added = True
        if not added:
            break

    return diversified


def get_families() -> list[str]:
    return sorted(f.value for f in ModelFamily)


def get_by_repo(repo_id: str, *, token: str | None = None) -> CatalogEntry | None:
    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError

    from seiso.models.hub_errors import is_hub_transport_error

    try:
        info = HfApi(token=token).model_info(repo_id)
    except HfHubHTTPError:
        return None
    except Exception as exc:
        if is_hub_transport_error(exc):
            return None
        raise
    return _hub_row_to_entry(_model_info_to_row(info))


def get_by_gguf_mirror(
    mirror_repo: str, *, token: str | None = None
) -> CatalogEntry | None:
    return get_by_repo(mirror_repo, token=token)
