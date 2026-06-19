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
    MINIMAX = "minimax"
    NEMOTRON = "nemotron"
    GLM = "glm"
    IBM = "ibm"
    OLMO = "olmo"
    LLAVA = "llava"
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


# Curated catalog — popular GGUF-friendly open-weight models for local use (June 2026)
CATALOG: tuple[CatalogEntry, ...] = (
    # ── Current flagships and high-signal local picks ──
    CatalogEntry(
        "Qwen/Qwen3.6-35B-A3B", "Qwen 3.6 35B MoE", ModelFamily.QWEN, "35B", ModelTask.CODE, "Q4_K_M",
        ("code", "moe", "new", "popular"), priority=100, gguf_repo="unsloth/Qwen3.6-35B-A3B-GGUF",
    ),
    CatalogEntry(
        "Qwen/Qwen3-Coder-Next", "Qwen 3 Coder Next", ModelFamily.QWEN, "80B", ModelTask.CODE, "Q4_K_M",
        ("code", "moe", "new", "popular"), priority=99, gguf_repo="unsloth/Qwen3-Coder-Next-GGUF",
    ),
    CatalogEntry(
        "openai/gpt-oss-20b", "GPT-OSS 20B", ModelFamily.OTHER, "20B", ModelTask.CHAT, "Q4_K_M",
        ("reasoning", "new", "popular"), priority=98, gguf_repo="unsloth/gpt-oss-20b-GGUF",
    ),
    CatalogEntry(
        "google/gemma-3-27b-it", "Gemma 3 27B Instruct", ModelFamily.GEMMA, "27B", ModelTask.CHAT, "Q4_K_M",
        ("vision", "new", "popular"), priority=97, gguf_repo="unsloth/gemma-3-27b-it-GGUF",
    ),
    CatalogEntry(
        "deepseek-ai/DeepSeek-R1-0528", "DeepSeek R1 0528", ModelFamily.DEEPSEEK, "671B", ModelTask.CHAT, "Q4_K_M",
        ("reasoning", "moe", "new", "popular"), priority=96, gguf_repo="unsloth/DeepSeek-R1-0528-GGUF",
    ),
    CatalogEntry(
        "meta-llama/Llama-4-Scout-17B-16E-Instruct", "Llama 4 Scout", ModelFamily.LLAMA, "109B", ModelTask.CHAT, "Q4_K_M",
        ("moe", "vision", "long-context", "popular"), priority=95, gguf_repo="unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF",
    ),
    CatalogEntry(
        "mistralai/Devstral-Small-2507", "Devstral Small 1.1", ModelFamily.MISTRAL, "24B", ModelTask.CODE, "Q4_K_M",
        ("code", "agent", "new", "popular"), priority=94,
    ),
    CatalogEntry(
        "Qwen/Qwen3-Coder-30B-A3B-Instruct", "Qwen 3 Coder 30B MoE", ModelFamily.QWEN, "30B", ModelTask.CODE, "Q4_K_M",
        ("code", "moe", "popular"), priority=93,
    ),
    CatalogEntry(
        "Qwen/Qwen3-30B-A3B", "Qwen 3 30B MoE", ModelFamily.QWEN, "30B", ModelTask.CHAT, "Q4_K_M",
        ("moe", "reasoning", "popular"), priority=92, gguf_repo="unsloth/Qwen3-30B-A3B-GGUF",
    ),
    CatalogEntry(
        "mistralai/Mistral-Small-3.2-24B-Instruct-2506", "Mistral Small 3.2 24B", ModelFamily.MISTRAL, "24B", ModelTask.CHAT, "Q4_K_M",
        ("vision", "new", "popular"), priority=91,
    ),
    CatalogEntry(
        "microsoft/phi-4", "Phi-4", ModelFamily.PHI, "14B", ModelTask.CHAT, "Q4_K_M",
        ("popular",), priority=90, gguf_repo="unsloth/phi-4-GGUF",
    ),
    CatalogEntry(
        "meta-llama/Llama-3.3-70B-Instruct", "Llama 3.3 70B Instruct", ModelFamily.LLAMA, "70B", ModelTask.CHAT, "Q4_K_M",
        ("popular",), priority=89, gguf_repo="unsloth/Llama-3.3-70B-Instruct-GGUF",
    ),
    CatalogEntry(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", "DeepSeek R1 Distill Qwen 32B", ModelFamily.DEEPSEEK, "32B", ModelTask.CHAT, "Q4_K_M",
        ("reasoning", "popular"), priority=88, gguf_repo="unsloth/DeepSeek-R1-Distill-Qwen-32B-GGUF",
    ),
    CatalogEntry(
        "Qwen/QwQ-32B", "QwQ 32B Reasoning", ModelFamily.QWEN, "32B", ModelTask.CHAT, "Q4_K_M",
        ("reasoning", "popular"), priority=87, gguf_repo="unsloth/QwQ-32B-GGUF",
    ),
    # ── Practical single-GPU and laptop-friendly models ──
    CatalogEntry(
        "Qwen/Qwen3-14B", "Qwen 3 14B", ModelFamily.QWEN, "14B", ModelTask.CHAT, "Q4_K_M",
        ("reasoning", "popular"), priority=86, gguf_repo="unsloth/Qwen3-14B-GGUF",
    ),
    CatalogEntry(
        "meta-llama/Llama-3.2-3B-Instruct", "Llama 3.2 3B Instruct", ModelFamily.LLAMA, "3B", ModelTask.CHAT, "Q4_K_M",
        ("small", "popular"), priority=85, gguf_repo="unsloth/Llama-3.2-3B-Instruct-GGUF",
    ),
    CatalogEntry(
        "google/gemma-3-12b-it", "Gemma 3 12B Instruct", ModelFamily.GEMMA, "12B", ModelTask.CHAT, "Q4_K_M",
        ("vision", "popular"), priority=85, gguf_repo="unsloth/gemma-3-12b-it-GGUF",
    ),
    CatalogEntry(
        "Qwen/Qwen3-8B", "Qwen 3 8B", ModelFamily.QWEN, "8B", ModelTask.CHAT, "Q4_K_M",
        ("reasoning", "popular"), priority=84, gguf_repo="unsloth/Qwen3-8B-GGUF",
    ),
    CatalogEntry(
        "google/gemma-3-4b-it", "Gemma 3 4B Instruct", ModelFamily.GEMMA, "4B", ModelTask.CHAT, "Q4_K_M",
        ("vision", "small", "popular"), priority=83, gguf_repo="unsloth/gemma-3-4b-it-GGUF",
    ),
    CatalogEntry(
        "microsoft/Phi-4-mini-instruct", "Phi-4 Mini Instruct", ModelFamily.PHI, "3.8B", ModelTask.CHAT, "Q4_K_M",
        ("small", "popular"), priority=82, gguf_repo="unsloth/Phi-4-mini-instruct-GGUF",
    ),
    CatalogEntry(
        "Qwen/Qwen3-4B", "Qwen 3 4B", ModelFamily.QWEN, "4B", ModelTask.CHAT, "Q4_K_M",
        ("small", "popular"), priority=81, gguf_repo="unsloth/Qwen3-4B-GGUF",
    ),
    CatalogEntry(
        "Qwen/Qwen3-1.7B", "Qwen 3 1.7B", ModelFamily.QWEN, "1.7B", ModelTask.CHAT, "Q4_K_M",
        ("small",), priority=80, gguf_repo="unsloth/Qwen3-1.7B-GGUF",
    ),
    CatalogEntry(
        "Qwen/Qwen3-0.6B", "Qwen 3 0.6B", ModelFamily.QWEN, "0.6B", ModelTask.CHAT, "Q4_K_M",
        ("small",), priority=79, gguf_repo="unsloth/Qwen3-0.6B-GGUF",
    ),
    CatalogEntry(
        "mistralai/Mistral-7B-Instruct-v0.3", "Mistral 7B Instruct v0.3", ModelFamily.MISTRAL, "7B", ModelTask.CHAT, "Q4_K_M",
        ("popular",), priority=78, gguf_repo="unsloth/Mistral-7B-Instruct-v0.3-GGUF",
    ),
    CatalogEntry(
        "google/gemma-3-1b-it", "Gemma 3 1B Instruct", ModelFamily.GEMMA, "1B", ModelTask.CHAT, "Q4_K_M",
        ("small",), priority=78, gguf_repo="unsloth/gemma-3-1b-it-GGUF",
    ),
    # ── Popular reasoning, coding, and high-end alternatives ──
    CatalogEntry(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", "DeepSeek R1 Distill Qwen 14B", ModelFamily.DEEPSEEK, "14B", ModelTask.CHAT, "Q4_K_M",
        ("reasoning", "popular"), priority=77,
    ),
    CatalogEntry(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "DeepSeek R1 Distill Qwen 7B", ModelFamily.DEEPSEEK, "7B", ModelTask.CHAT, "Q4_K_M",
        ("reasoning", "popular"), priority=76,
    ),
    CatalogEntry(
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "DeepSeek R1 Distill Llama 8B", ModelFamily.DEEPSEEK, "8B", ModelTask.CHAT, "Q4_K_M",
        ("reasoning",), priority=75,
    ),
    CatalogEntry(
        "mistralai/Codestral-22B-v0.1", "Codestral 22B", ModelFamily.MISTRAL, "22B", ModelTask.CODE, "Q4_K_M",
        ("code",), priority=74,
    ),
    CatalogEntry(
        "mistralai/Mistral-Small-3.1-24B-Instruct-2503", "Mistral Small 3.1 24B", ModelFamily.MISTRAL, "24B", ModelTask.CHAT, "Q4_K_M",
        ("vision", "popular"), priority=73,
    ),
    CatalogEntry(
        "moonshotai/Kimi-K2-Instruct", "Kimi K2 Instruct", ModelFamily.KIMI, "1T", ModelTask.CHAT, "Q4_K_M",
        ("moe", "code", "popular"), priority=72,
    ),
    CatalogEntry(
        "zai-org/GLM-4.5-Air", "GLM 4.5 Air", ModelFamily.GLM, "106B", ModelTask.CHAT, "Q4_K_M",
        ("agent", "moe", "popular"), priority=71,
    ),
    CatalogEntry(
        "openai/gpt-oss-120b", "GPT-OSS 120B", ModelFamily.OTHER, "120B", ModelTask.CHAT, "Q4_K_M",
        ("reasoning", "moe", "popular"), priority=70, gguf_repo="unsloth/gpt-oss-120b-GGUF",
    ),
    CatalogEntry(
        "meta-llama/Llama-4-Maverick-17B-128E-Instruct", "Llama 4 Maverick", ModelFamily.LLAMA, "400B", ModelTask.CHAT, "Q4_K_M",
        ("moe", "vision"), priority=69,
    ),
    CatalogEntry(
        "deepseek-ai/DeepSeek-V3-0324", "DeepSeek V3 0324", ModelFamily.DEEPSEEK, "671B", ModelTask.CHAT, "Q4_K_M",
        ("moe", "popular"), priority=68,
    ),
    CatalogEntry(
        "deepseek-ai/DeepSeek-R1-Distill-Llama-70B", "DeepSeek R1 Distill Llama 70B", ModelFamily.DEEPSEEK, "70B", ModelTask.CHAT, "Q4_K_M",
        ("reasoning",), priority=67,
    ),
    # ── Vision-language models ──
    CatalogEntry(
        "Qwen/Qwen3-VL-2B-Instruct", "Qwen 3 VL 2B", ModelFamily.QWEN, "2B", ModelTask.VISION, "Q4_K_M",
        ("vision", "small", "popular"), priority=66,
    ),
    CatalogEntry(
        "Qwen/Qwen3-VL-8B-Instruct", "Qwen 3 VL 8B", ModelFamily.QWEN, "8B", ModelTask.VISION, "Q4_K_M",
        ("vision", "popular"), priority=65,
    ),
    CatalogEntry(
        "Qwen/Qwen3-VL-4B-Instruct", "Qwen 3 VL 4B", ModelFamily.QWEN, "4B", ModelTask.VISION, "Q4_K_M",
        ("vision",), priority=64,
    ),
    # ── Embeddings ──
    CatalogEntry("BAAI/bge-small-en-v1.5", "BGE Small EN", ModelFamily.OTHER, "33M", ModelTask.EMBEDDING, "F16", priority=20),
    CatalogEntry("BAAI/bge-m3", "BGE M3 Multilingual", ModelFamily.OTHER, "568M", ModelTask.EMBEDDING, "F16", priority=19),
    CatalogEntry("sentence-transformers/all-MiniLM-L6-v2", "MiniLM L6 v2", ModelFamily.OTHER, "22M", ModelTask.EMBEDDING, "F16", priority=18),
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
