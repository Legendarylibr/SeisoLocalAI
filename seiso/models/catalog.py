"""Popular Hugging Face models curated for local training and inference."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from seiso.compat import StrEnum


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
    priority: int = 50  # higher = shown first (newer / more relevant)


# Curated catalog — Qwen 3.6 / 3.5 and recent open-weight models (June 2026)
CATALOG: tuple[CatalogEntry, ...] = (
    # ── Qwen 3.6 (current generation) ──
    CatalogEntry(
        "Qwen/Qwen3.6-35B-A3B", "Qwen 3.6 35B MoE", ModelFamily.QWEN, "35B", ModelTask.CODE, "Q4_K_M",
        ("code", "moe", "new", "popular"), priority=100, gguf_repo="unsloth/Qwen3.6-35B-A3B-GGUF",
    ),
    CatalogEntry(
        "Qwen/Qwen3.6-27B", "Qwen 3.6 27B", ModelFamily.QWEN, "27B", ModelTask.CHAT, "Q4_K_M",
        ("reasoning", "new", "popular"), priority=99, gguf_repo="unsloth/Qwen3.6-27B-GGUF",
    ),
    CatalogEntry(
        "Qwen/Qwen3.6-9B", "Qwen 3.6 9B", ModelFamily.QWEN, "9B", ModelTask.CHAT, "Q4_K_M",
        ("reasoning", "new", "popular"), priority=98, gguf_repo="unsloth/Qwen3.6-9B-GGUF",
    ),
    CatalogEntry(
        "Qwen/Qwen3.6-4B", "Qwen 3.6 4B", ModelFamily.QWEN, "4B", ModelTask.CHAT, "Q4_K_M",
        ("small", "new", "popular"), priority=97, gguf_repo="unsloth/Qwen3.6-4B-GGUF",
    ),
    CatalogEntry(
        "Qwen/Qwen3.6-1.7B", "Qwen 3.6 1.7B", ModelFamily.QWEN, "1.7B", ModelTask.CHAT, "Q4_K_M",
        ("small", "new"), priority=96, gguf_repo="unsloth/Qwen3.6-1.7B-GGUF",
    ),
    # ── Qwen 3.5 (widely available GGUF mirrors) ──
    CatalogEntry(
        "Qwen/Qwen3.5-27B", "Qwen 3.5 27B", ModelFamily.QWEN, "27B", ModelTask.CHAT, "Q4_K_M",
        ("reasoning", "popular"), priority=95, gguf_repo="unsloth/Qwen3.5-27B-GGUF",
    ),
    CatalogEntry(
        "Qwen/Qwen3.5-4B", "Qwen 3.5 4B", ModelFamily.QWEN, "4B", ModelTask.CHAT, "Q4_K_M",
        ("small", "popular"), priority=94, gguf_repo="unsloth/Qwen3.5-4B-GGUF",
    ),
    CatalogEntry(
        "Qwen/Qwen3.5-2B", "Qwen 3.5 2B", ModelFamily.QWEN, "2B", ModelTask.CHAT, "Q4_K_M",
        ("small", "popular"), priority=93, gguf_repo="unsloth/Qwen3.5-2B-GGUF",
    ),
    # ── Qwen coding ──
    CatalogEntry(
        "Qwen/Qwen3-Coder-Next", "Qwen 3 Coder Next", ModelFamily.QWEN, "80B", ModelTask.CODE, "Q4_K_M",
        ("code", "moe", "new", "popular"), priority=92, gguf_repo="unsloth/Qwen3-Coder-Next-GGUF",
    ),
    CatalogEntry(
        "Qwen/Qwen3-Coder-30B-A3B-Instruct", "Qwen 3 Coder 30B MoE", ModelFamily.QWEN, "30B", ModelTask.CODE, "Q4_K_M",
        ("code", "moe", "popular"), priority=91, gguf_repo="unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF",
    ),
    # ── Recent flagship chat / reasoning models ──
    CatalogEntry(
        "openai/gpt-oss-20b", "GPT-OSS 20B", ModelFamily.OTHER, "20B", ModelTask.CHAT, "Q4_K_M",
        ("reasoning", "new", "popular"), priority=90, gguf_repo="unsloth/gpt-oss-20b-GGUF",
    ),
    CatalogEntry(
        "openai/gpt-oss-120b", "GPT-OSS 120B", ModelFamily.OTHER, "120B", ModelTask.CHAT, "Q4_K_M",
        ("reasoning", "moe", "popular"), priority=89, gguf_repo="unsloth/gpt-oss-120b-GGUF",
    ),
    CatalogEntry(
        "deepseek-ai/DeepSeek-R1-0528", "DeepSeek R1 0528", ModelFamily.DEEPSEEK, "671B", ModelTask.CHAT, "Q4_K_M",
        ("reasoning", "moe", "new", "popular"), priority=88, gguf_repo="unsloth/DeepSeek-R1-0528-GGUF",
    ),
    CatalogEntry(
        "meta-llama/Llama-4-Scout-17B-16E-Instruct", "Llama 4 Scout", ModelFamily.LLAMA, "109B", ModelTask.CHAT, "Q4_K_M",
        ("moe", "vision", "long-context", "popular"), priority=87, gguf_repo="unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF",
    ),
    CatalogEntry(
        "meta-llama/Llama-4-Maverick-17B-128E-Instruct", "Llama 4 Maverick", ModelFamily.LLAMA, "400B", ModelTask.CHAT, "Q4_K_M",
        ("moe", "vision", "new"), priority=86, gguf_repo="unsloth/Llama-4-Maverick-17B-128E-Instruct-GGUF",
    ),
    CatalogEntry(
        "google/gemma-3-27b-it", "Gemma 3 27B Instruct", ModelFamily.GEMMA, "27B", ModelTask.CHAT, "Q4_K_M",
        ("vision", "popular"), priority=85, gguf_repo="unsloth/gemma-3-27b-it-GGUF",
    ),
    CatalogEntry(
        "google/gemma-3-12b-it", "Gemma 3 12B Instruct", ModelFamily.GEMMA, "12B", ModelTask.CHAT, "Q4_K_M",
        ("vision", "popular"), priority=84, gguf_repo="unsloth/gemma-3-12b-it-GGUF",
    ),
    CatalogEntry(
        "google/gemma-3-4b-it", "Gemma 3 4B Instruct", ModelFamily.GEMMA, "4B", ModelTask.CHAT, "Q4_K_M",
        ("vision", "small", "popular"), priority=83, gguf_repo="unsloth/gemma-3-4b-it-GGUF",
    ),
    CatalogEntry(
        "mistralai/Devstral-Small-2507", "Devstral Small 1.1", ModelFamily.MISTRAL, "24B", ModelTask.CODE, "Q4_K_M",
        ("code", "agent", "new", "popular"), priority=82, gguf_repo="bartowski/Devstral-Small-2507-GGUF",
    ),
    CatalogEntry(
        "mistralai/Mistral-Small-3.2-24B-Instruct-2506", "Mistral Small 3.2 24B", ModelFamily.MISTRAL, "24B", ModelTask.CHAT, "Q4_K_M",
        ("vision", "new", "popular"), priority=81, gguf_repo="bartowski/Mistral-Small-3.2-24B-Instruct-2506-GGUF",
    ),
    CatalogEntry(
        "moonshotai/Kimi-K2-Instruct", "Kimi K2 Instruct", ModelFamily.KIMI, "1T", ModelTask.CHAT, "Q4_K_M",
        ("moe", "code", "popular"), priority=80, gguf_repo="unsloth/Kimi-K2-Instruct-GGUF",
    ),
    CatalogEntry(
        "moonshotai/Kimi-K2.7-Code", "Kimi K2.7 Code", ModelFamily.KIMI, "1T", ModelTask.CODE, "Q4_K_M",
        ("code", "moe", "new", "popular"), priority=79, gguf_repo="AesSedai/Kimi-K2.7-Code-GGUF",
    ),
    CatalogEntry(
        "zai-org/GLM-4.5-Air", "GLM 4.5 Air", ModelFamily.GLM, "106B", ModelTask.CHAT, "Q4_K_M",
        ("agent", "moe", "popular"), priority=78, gguf_repo="unsloth/GLM-4.5-Air-GGUF",
    ),
    CatalogEntry(
        "microsoft/Phi-4-mini-instruct", "Phi-4 Mini Instruct", ModelFamily.PHI, "3.8B", ModelTask.CHAT, "Q4_K_M",
        ("small", "popular"), priority=77, gguf_repo="unsloth/Phi-4-mini-instruct-GGUF",
    ),
    # ── Vision-language (Qwen 3.6 VL when available; Qwen3 VL as fallback) ──
    CatalogEntry(
        "Qwen/Qwen3-VL-8B-Instruct", "Qwen 3 VL 8B", ModelFamily.QWEN, "8B", ModelTask.VISION, "Q4_K_M",
        ("vision", "popular"), priority=70,
    ),
    CatalogEntry(
        "Qwen/Qwen3-VL-4B-Instruct", "Qwen 3 VL 4B", ModelFamily.QWEN, "4B", ModelTask.VISION, "Q4_K_M",
        ("vision",), priority=69,
    ),
    CatalogEntry(
        "Qwen/Qwen3-VL-2B-Instruct", "Qwen 3 VL 2B", ModelFamily.QWEN, "2B", ModelTask.VISION, "Q4_K_M",
        ("vision", "small"), priority=68,
    ),
    # ── Embeddings (for knowledge / RAG) ──
    CatalogEntry("BAAI/bge-small-en-v1.5", "BGE Small EN", ModelFamily.OTHER, "33M", ModelTask.EMBEDDING, "F16", priority=20),
    CatalogEntry("BAAI/bge-m3", "BGE M3 Multilingual", ModelFamily.OTHER, "568M", ModelTask.EMBEDDING, "F16", priority=19),
)


def _parse_param_size(params: str) -> float:
    p = params.strip().upper()
    if p.endswith("T"):
        return float(p[:-1]) * 1000
    if p.endswith("B"):
        return float(p[:-1])
    if p.endswith("M"):
        return float(p[:-1]) / 1000
    return float(p)


def _entry_to_dict(e: CatalogEntry) -> dict:
    return {
        "repo_id": e.repo_id,
        "name": e.name,
        "family": e.family.value,
        "params": e.params,
        "task": e.task.value,
        "quant": e.quant,
        "tags": list(e.tags),
        "gguf_repo": e.gguf_repo,
        "featured": e.priority >= 87,
        "priority": e.priority,
    }


def _matches_task(entry: CatalogEntry, task: str) -> bool:
    """Task filter — chat includes general instruct models tagged for code use."""
    task_l = task.lower()
    return (
        entry.task.value == task_l
        or (task_l == "chat" and entry.task == ModelTask.CODE and "popular" in entry.tags)
    )


def search_catalog(
    query: str = "",
    family: str | None = None,
    task: str | None = None,
    max_params: str | None = None,
) -> list[dict]:
    """Search and rank catalog entries — newer models rise when query is empty."""
    q = query.lower().strip()
    scored: list[tuple[float, dict]] = []

    for e in CATALOG:
        if family and e.family.value != family.lower():
            continue
        if task and not _matches_task(e, task):
            continue
        if max_params and _parse_param_size(e.params) > _parse_param_size(max_params):
            continue

        score = _relevance_score(e, q)
        if score < 0:
            continue
        scored.append((score, _entry_to_dict(e)))

    scored.sort(key=lambda pair: (-pair[0], pair[1]["name"]))
    return [item for _, item in scored]


def diversify_by_family(models: list[dict]) -> list[dict]:
    """Interleave families so the default catalog is not dominated by one brand."""
    if len(models) <= 1:
        return list(models)

    by_family: dict[str, list[dict]] = defaultdict(list)
    for m in models:
        by_family[m.get("family") or "other"].append(m)

    family_order = sorted(
        by_family.keys(),
        key=lambda fam: -(by_family[fam][0].get("priority") or 0) if by_family[fam] else 0,
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


def _relevance_score(entry: CatalogEntry, query: str) -> float:
    """Higher = better match. Returns -1 when query tokens miss entirely."""
    base = float(entry.priority)
    if not query:
        return base

    tokens = [t for t in query.split() if t]
    if not tokens:
        return base

    name_l = entry.name.lower()
    repo_l = entry.repo_id.lower()
    haystack = f"{name_l} {repo_l} {' '.join(entry.tags)} {entry.family.value} {entry.task.value} {entry.params.lower()}"

    score = base
    for tok in tokens:
        if tok in repo_l:
            score += 40
        elif name_l.startswith(tok):
            score += 35
        elif tok in name_l:
            score += 22
        elif tok in haystack:
            score += 12
        else:
            if _subsequence(tok, name_l) or _subsequence(tok, repo_l):
                score += 6
            else:
                return -1

    if "new" in entry.tags:
        score += 8
    if entry.priority >= 87:
        score += 5
    return score


def _subsequence(needle: str, haystack: str) -> bool:
    it = iter(haystack)
    return all(c in it for c in needle)


def get_families() -> list[str]:
    return sorted({e.family.value for e in CATALOG})


_BY_REPO_ID: dict[str, CatalogEntry] = {e.repo_id: e for e in CATALOG}
_BY_GGUF_REPO: dict[str, CatalogEntry] = {
    e.gguf_repo: e for e in CATALOG if e.gguf_repo is not None
}


def get_by_repo(repo_id: str) -> CatalogEntry | None:
    return _BY_REPO_ID.get(repo_id)


def get_by_gguf_mirror(mirror_repo: str) -> CatalogEntry | None:
    """Map a GGUF mirror repo back to its catalog base model."""
    return _BY_GGUF_REPO.get(mirror_repo)
