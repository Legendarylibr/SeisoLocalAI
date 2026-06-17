"""Popular Hugging Face models curated for local training and inference."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


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


# Curated catalog — ranked by Hugging Face download trends (June 2026)
CATALOG: tuple[CatalogEntry, ...] = (
    # ── Top coding / agentic ──
    CatalogEntry("moonshotai/Kimi-K2.7-Code", "Kimi K2.7 Code", ModelFamily.KIMI, "1T", ModelTask.CODE, "Q4_K_M", ("code", "moe", "new", "popular"), priority=100),
    CatalogEntry("Qwen/Qwen3.6-35B-A3B", "Qwen 3.6 35B MoE", ModelFamily.QWEN, "35B", ModelTask.CODE, "Q4_K_M", ("code", "moe", "new", "popular"), priority=99),
    CatalogEntry("moonshotai/Kimi-K2.6", "Kimi K2.6", ModelFamily.KIMI, "1T", ModelTask.CODE, "Q4_K_M", ("code", "moe", "new", "popular"), priority=98),
    CatalogEntry("Qwen/Qwen3.6-27B", "Qwen 3.6 27B", ModelFamily.QWEN, "27B", ModelTask.CODE, "Q4_K_M", ("code", "new", "popular"), priority=97),
    CatalogEntry("nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16", "Nemotron 3 Super 120B", ModelFamily.NEMOTRON, "120B", ModelTask.CHAT, "Q4_K_M", ("moe", "new", "popular", "reasoning"), priority=96),
    CatalogEntry("zai-org/GLM-5", "GLM 5", ModelFamily.OTHER, "744B", ModelTask.CHAT, "Q4_K_M", ("moe", "new", "popular", "reasoning"), priority=95),
    CatalogEntry("Qwen/Qwen3-Coder-Next", "Qwen 3 Coder Next", ModelFamily.QWEN, "80B", ModelTask.CODE, "Q4_K_M", ("code", "moe", "new"), priority=94),
    CatalogEntry("moonshotai/Kimi-K2.5", "Kimi K2.5", ModelFamily.KIMI, "1T", ModelTask.CHAT, "Q4_K_M", ("moe", "new", "popular", "vision"), priority=93),
    CatalogEntry("moonshotai/Kimi-K2-Thinking", "Kimi K2 Thinking", ModelFamily.KIMI, "1T", ModelTask.CHAT, "Q4_K_M", ("moe", "new", "reasoning"), priority=92),
    CatalogEntry("MiniMaxAI/MiniMax-M2.7", "MiniMax M2.7", ModelFamily.MINIMAX, "230B", ModelTask.CHAT, "Q4_K_M", ("moe", "new", "popular"), priority=91),
    CatalogEntry("google/gemma-4-31B-it", "Gemma 4 31B Instruct", ModelFamily.GEMMA, "31B", ModelTask.CODE, "Q4_K_M", ("code", "new"), priority=90),
    CatalogEntry("google/gemma-4-26B-A4B-it", "Gemma 4 26B MoE", ModelFamily.GEMMA, "26B", ModelTask.CODE, "Q4_K_M", ("code", "moe", "new"), priority=89),
    CatalogEntry("MiniMaxAI/MiniMax-M3", "MiniMax M3", ModelFamily.MINIMAX, "23B", ModelTask.CODE, "Q4_K_M", ("code", "moe", "new"), priority=88),
    CatalogEntry("zai-org/GLM-4.7", "GLM 4.7", ModelFamily.OTHER, "355B", ModelTask.CODE, "Q4_K_M", ("code", "new"), priority=87),
    CatalogEntry("mistralai/Devstral-Small-2507", "Devstral Small 1.1", ModelFamily.MISTRAL, "24B", ModelTask.CODE, "Q4_K_M", ("code", "new"), priority=86),
    CatalogEntry("Qwen/Qwen3-Coder-30B-A3B-Instruct", "Qwen 3 Coder 30B MoE", ModelFamily.QWEN, "30B", ModelTask.CODE, "Q4_K_M", ("code", "moe"), priority=85),
    CatalogEntry("google/gemma-4-12B-it", "Gemma 4 12B Instruct", ModelFamily.GEMMA, "12B", ModelTask.CODE, "Q4_K_M", ("code", "new", "vision"), priority=84),
    CatalogEntry("google/gemma-4-E4B-it", "Gemma 4 E4B Instruct", ModelFamily.GEMMA, "4B", ModelTask.CODE, "Q4_K_M", ("code", "new"), priority=83),
    CatalogEntry("google/gemma-4-E2B-it", "Gemma 4 E2B Instruct", ModelFamily.GEMMA, "2B", ModelTask.CODE, "Q4_K_M", ("code", "new"), priority=82),
    CatalogEntry("zai-org/GLM-4.7-Flash", "GLM 4.7 Flash", ModelFamily.OTHER, "30B", ModelTask.CODE, "Q4_K_M", ("code", "moe"), priority=81),
    CatalogEntry("meta-llama/Llama-4-Scout-17B-16E-Instruct", "Llama 4 Scout", ModelFamily.LLAMA, "109B", ModelTask.CHAT, "Q4_K_M", ("moe", "vision", "new"), priority=80),
    CatalogEntry("openai/gpt-oss-20b", "GPT-OSS 20B", ModelFamily.OTHER, "20B", ModelTask.CHAT, "Q4_K_M", ("new", "popular"), priority=79),
    CatalogEntry("Qwen/Qwen3.5-35B-A3B", "Qwen 3.5 35B MoE", ModelFamily.QWEN, "35B", ModelTask.CHAT, "Q4_K_M", ("moe", "popular"), priority=78),
    CatalogEntry("Qwen/Qwen3.5-27B", "Qwen 3.5 27B", ModelFamily.QWEN, "27B", ModelTask.CHAT, "Q4_K_M", ("popular",), priority=77),
    CatalogEntry("Qwen/Qwen3.5-9B", "Qwen 3.5 9B", ModelFamily.QWEN, "9B", ModelTask.CHAT, "Q4_K_M", ("popular",), priority=76),
    CatalogEntry("Qwen/Qwen3.5-4B", "Qwen 3.5 4B", ModelFamily.QWEN, "4B", ModelTask.CHAT, "Q4_K_M", ("popular",), priority=75),
    CatalogEntry("deepseek-ai/DeepSeek-R1-0528", "DeepSeek R1 (May 2025)", ModelFamily.DEEPSEEK, "671B", ModelTask.CHAT, "Q4_K_M", ("reasoning", "moe", "popular"), priority=74),
    CatalogEntry("deepseek-ai/DeepSeek-V3.2", "DeepSeek V3.2", ModelFamily.DEEPSEEK, "671B", ModelTask.CHAT, "Q4_K_M", ("moe",), priority=73),
    CatalogEntry("microsoft/phi-4", "Phi-4", ModelFamily.PHI, "14B", ModelTask.CHAT, "Q4_K_M", priority=72),
    CatalogEntry("microsoft/Phi-4-mini-instruct", "Phi-4 Mini", ModelFamily.PHI, "3.8B", ModelTask.CHAT, "Q4_K_M", priority=71),
    CatalogEntry("mistralai/Codestral-22B-v0.1", "Codestral 22B", ModelFamily.MISTRAL, "22B", ModelTask.CODE, "Q4_K_M", ("code",), priority=70),
    CatalogEntry("mistralai/Mistral-Small-4-119B-2603", "Mistral Small 4", ModelFamily.MISTRAL, "119B", ModelTask.CHAT, "Q4_K_M", ("moe", "new"), priority=69),
    CatalogEntry("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "DeepSeek R1 Distill 8B", ModelFamily.DEEPSEEK, "8B", ModelTask.CHAT, "Q4_K_M", ("reasoning",), priority=68),
    CatalogEntry("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "DeepSeek R1 Distill Qwen 7B", ModelFamily.DEEPSEEK, "7B", ModelTask.CHAT, "Q4_K_M", ("reasoning",), priority=67),
    CatalogEntry("Qwen/Qwen3.5-122B-A10B", "Qwen 3.5 122B MoE", ModelFamily.QWEN, "122B", ModelTask.CHAT, "Q4_K_M", ("moe",), priority=66),
    CatalogEntry("Qwen/Qwen3.5-0.8B", "Qwen 3.5 0.8B", ModelFamily.QWEN, "0.8B", ModelTask.CHAT, "Q4_K_M", ("popular",), priority=65),
    CatalogEntry("Qwen/Qwen3.5-2B", "Qwen 3.5 2B", ModelFamily.QWEN, "2B", ModelTask.CHAT, "Q4_K_M", priority=64),
    # ── Vision ──
    CatalogEntry("Qwen/Qwen3-VL-2B-Instruct", "Qwen 3 VL 2B", ModelFamily.QWEN, "2B", ModelTask.VISION, "Q4_K_M", ("popular",), priority=63),
    CatalogEntry("Qwen/Qwen3-VL-8B-Instruct", "Qwen 3 VL 8B", ModelFamily.QWEN, "8B", ModelTask.VISION, "Q4_K_M", priority=62),
    CatalogEntry("Qwen/Qwen3-VL-4B-Instruct", "Qwen 3 VL 4B", ModelFamily.QWEN, "4B", ModelTask.VISION, "Q4_K_M", priority=61),
    # ── Frontier / datacenter ──
    CatalogEntry("nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16", "Nemotron 3 Ultra 550B", ModelFamily.NEMOTRON, "550B", ModelTask.CHAT, "Q4_K_M", ("moe", "reasoning"), priority=50),
    CatalogEntry("deepseek-ai/DeepSeek-V4-Flash", "DeepSeek V4 Flash", ModelFamily.DEEPSEEK, "284B", ModelTask.CHAT, "Q4_K_M", ("moe",), priority=49),
    CatalogEntry("deepseek-ai/DeepSeek-V4-Pro", "DeepSeek V4 Pro", ModelFamily.DEEPSEEK, "1.6T", ModelTask.CHAT, "Q4_K_M", ("moe",), priority=48),
    CatalogEntry("Qwen/Qwen3.5-397B-A17B", "Qwen 3.5 397B MoE", ModelFamily.QWEN, "397B", ModelTask.CHAT, "Q4_K_M", ("moe",), priority=47),
    CatalogEntry("openai/gpt-oss-120b", "GPT-OSS 120B", ModelFamily.OTHER, "120B", ModelTask.CHAT, "Q4_K_M", ("popular",), priority=46),
    # ── Embeddings ──
    CatalogEntry("BAAI/bge-small-en-v1.5", "BGE Small EN", ModelFamily.OTHER, "33M", ModelTask.EMBEDDING, "F16", priority=20),
    CatalogEntry("BAAI/bge-m3", "BGE M3 Multilingual", ModelFamily.OTHER, "568M", ModelTask.EMBEDDING, "F16", priority=19),
    CatalogEntry("sentence-transformers/all-MiniLM-L6-v2", "MiniLM L6 v2", ModelFamily.OTHER, "22M", ModelTask.EMBEDDING, "F16", priority=18),
)


def _parse_param_size(params: str) -> float:
    p = params.strip().upper()
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
        if task and e.task.value != task.lower():
            continue
        if max_params and _parse_param_size(e.params) > _parse_param_size(max_params):
            continue

        score = _relevance_score(e, q)
        if score < 0:
            continue
        scored.append((score, _entry_to_dict(e)))

    scored.sort(key=lambda pair: (-pair[0], pair[1]["name"]))
    return [item for _, item in scored]


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
            # allow fuzzy: all chars of token appear in order in name
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


def get_by_repo(repo_id: str) -> CatalogEntry | None:
    for e in CATALOG:
        if e.repo_id == repo_id:
            return e
    return None
