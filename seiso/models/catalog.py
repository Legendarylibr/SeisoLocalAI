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
    quant: str  # recommended quant for inference download
    tags: tuple[str, ...] = ()
    gguf_repo: str | None = None  # alternate GGUF repo if different


# Curated catalog — popular open models on Hugging Face
CATALOG: tuple[CatalogEntry, ...] = (
    # Llama 3.x
    CatalogEntry("meta-llama/Llama-3.2-1B-Instruct", "Llama 3.2 1B Instruct", ModelFamily.LLAMA, "1B", ModelTask.CHAT, "Q4_K_M"),
    CatalogEntry("meta-llama/Llama-3.2-3B-Instruct", "Llama 3.2 3B Instruct", ModelFamily.LLAMA, "3B", ModelTask.CHAT, "Q4_K_M"),
    CatalogEntry("meta-llama/Llama-3.1-8B-Instruct", "Llama 3.1 8B Instruct", ModelFamily.LLAMA, "8B", ModelTask.CHAT, "Q4_K_M"),
    CatalogEntry("meta-llama/Llama-3.1-70B-Instruct", "Llama 3.1 70B Instruct", ModelFamily.LLAMA, "70B", ModelTask.CHAT, "Q4_K_M"),
    CatalogEntry("meta-llama/Llama-3.2-11B-Vision-Instruct", "Llama 3.2 11B Vision", ModelFamily.LLAVA, "11B", ModelTask.VISION, "Q4_K_M"),
    # Qwen 2.5 / 3
    CatalogEntry("Qwen/Qwen2.5-0.5B-Instruct", "Qwen 2.5 0.5B Instruct", ModelFamily.QWEN, "0.5B", ModelTask.CHAT, "Q4_K_M"),
    CatalogEntry("Qwen/Qwen2.5-1.5B-Instruct", "Qwen 2.5 1.5B Instruct", ModelFamily.QWEN, "1.5B", ModelTask.CHAT, "Q4_K_M"),
    CatalogEntry("Qwen/Qwen2.5-3B-Instruct", "Qwen 2.5 3B Instruct", ModelFamily.QWEN, "3B", ModelTask.CHAT, "Q4_K_M"),
    CatalogEntry("Qwen/Qwen2.5-7B-Instruct", "Qwen 2.5 7B Instruct", ModelFamily.QWEN, "7B", ModelTask.CHAT, "Q4_K_M"),
    CatalogEntry("Qwen/Qwen2.5-14B-Instruct", "Qwen 2.5 14B Instruct", ModelFamily.QWEN, "14B", ModelTask.CHAT, "Q4_K_M"),
    CatalogEntry("Qwen/Qwen2.5-32B-Instruct", "Qwen 2.5 32B Instruct", ModelFamily.QWEN, "32B", ModelTask.CHAT, "Q4_K_M"),
    CatalogEntry("Qwen/Qwen2.5-72B-Instruct", "Qwen 2.5 72B Instruct", ModelFamily.QWEN, "72B", ModelTask.CHAT, "Q4_K_M"),
    CatalogEntry("Qwen/Qwen2.5-Coder-7B-Instruct", "Qwen 2.5 Coder 7B", ModelFamily.QWEN, "7B", ModelTask.CODE, "Q4_K_M", ("code",)),
    CatalogEntry("Qwen/Qwen2.5-Coder-32B-Instruct", "Qwen 2.5 Coder 32B", ModelFamily.QWEN, "32B", ModelTask.CODE, "Q4_K_M", ("code",)),
    CatalogEntry("Qwen/Qwen3-8B", "Qwen 3 8B", ModelFamily.QWEN, "8B", ModelTask.CHAT, "Q4_K_M"),
    CatalogEntry("Qwen/Qwen3-30B-A3B", "Qwen 3 30B MoE", ModelFamily.QWEN, "30B", ModelTask.CHAT, "Q4_K_M", ("moe",)),
    # Gemma
    CatalogEntry("google/gemma-2-2b-it", "Gemma 2 2B Instruct", ModelFamily.GEMMA, "2B", ModelTask.CHAT, "Q4_K_M"),
    CatalogEntry("google/gemma-2-9b-it", "Gemma 2 9B Instruct", ModelFamily.GEMMA, "9B", ModelTask.CHAT, "Q4_K_M"),
    CatalogEntry("google/gemma-2-27b-it", "Gemma 2 27B Instruct", ModelFamily.GEMMA, "27B", ModelTask.CHAT, "Q4_K_M"),
    CatalogEntry("google/gemma-3-4b-it", "Gemma 3 4B Instruct", ModelFamily.GEMMA, "4B", ModelTask.CHAT, "Q4_K_M"),
    CatalogEntry("google/gemma-3-12b-it", "Gemma 3 12B Instruct", ModelFamily.GEMMA, "12B", ModelTask.CHAT, "Q4_K_M"),
    CatalogEntry("google/gemma-3-27b-it", "Gemma 3 27B Instruct", ModelFamily.GEMMA, "27B", ModelTask.CHAT, "Q4_K_M"),
    # Phi
    CatalogEntry("microsoft/Phi-3-mini-4k-instruct", "Phi-3 Mini 4K", ModelFamily.PHI, "3.8B", ModelTask.CHAT, "Q4_K_M"),
    CatalogEntry("microsoft/Phi-3.5-mini-instruct", "Phi-3.5 Mini", ModelFamily.PHI, "3.8B", ModelTask.CHAT, "Q4_K_M"),
    CatalogEntry("microsoft/phi-4", "Phi-4", ModelFamily.PHI, "14B", ModelTask.CHAT, "Q4_K_M"),
    # Mistral / Mixtral
    CatalogEntry("mistralai/Mistral-7B-Instruct-v0.3", "Mistral 7B v0.3", ModelFamily.MISTRAL, "7B", ModelTask.CHAT, "Q4_K_M"),
    CatalogEntry("mistralai/Mixtral-8x7B-Instruct-v0.1", "Mixtral 8x7B", ModelFamily.MISTRAL, "47B", ModelTask.CHAT, "Q4_K_M", ("moe",)),
    CatalogEntry("mistralai/Mixtral-8x22B-Instruct-v0.1", "Mixtral 8x22B", ModelFamily.MISTRAL, "141B", ModelTask.CHAT, "Q4_K_M", ("moe",)),
    CatalogEntry("mistralai/Mistral-Small-Instruct-2409", "Mistral Small", ModelFamily.MISTRAL, "22B", ModelTask.CHAT, "Q4_K_M"),
    CatalogEntry("mistralai/Codestral-22B-v0.1", "Codestral 22B", ModelFamily.MISTRAL, "22B", ModelTask.CODE, "Q4_K_M", ("code",)),
    # DeepSeek
    CatalogEntry("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "DeepSeek R1 Distill 8B", ModelFamily.DEEPSEEK, "8B", ModelTask.CHAT, "Q4_K_M", ("reasoning",)),
    CatalogEntry("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "DeepSeek R1 Distill Qwen 7B", ModelFamily.DEEPSEEK, "7B", ModelTask.CHAT, "Q4_K_M", ("reasoning",)),
    CatalogEntry("deepseek-ai/DeepSeek-V2.5", "DeepSeek V2.5", ModelFamily.DEEPSEEK, "236B", ModelTask.CHAT, "Q4_K_M", ("moe",)),
    CatalogEntry("deepseek-ai/deepseek-coder-6.7b-instruct", "DeepSeek Coder 6.7B", ModelFamily.DEEPSEEK, "6.7B", ModelTask.CODE, "Q4_K_M", ("code",)),
    # Other popular
    CatalogEntry("01-ai/Yi-1.5-9B-Chat", "Yi 1.5 9B Chat", ModelFamily.OTHER, "9B", ModelTask.CHAT, "Q4_K_M"),
    CatalogEntry("01-ai/Yi-1.5-34B-Chat", "Yi 1.5 34B Chat", ModelFamily.OTHER, "34B", ModelTask.CHAT, "Q4_K_M"),
    CatalogEntry("tiiuae/falcon-7b-instruct", "Falcon 7B Instruct", ModelFamily.OTHER, "7B", ModelTask.CHAT, "Q4_K_M"),
    CatalogEntry("HuggingFaceTB/SmolLM2-1.7B-Instruct", "SmolLM2 1.7B", ModelFamily.OTHER, "1.7B", ModelTask.CHAT, "Q4_K_M"),
    CatalogEntry("HuggingFaceTB/SmolLM2-360M-Instruct", "SmolLM2 360M", ModelFamily.OTHER, "360M", ModelTask.CHAT, "Q4_K_M"),
    CatalogEntry("stabilityai/stablelm-2-zephyr-1_6b", "StableLM 2 Zephyr 1.6B", ModelFamily.OTHER, "1.6B", ModelTask.CHAT, "Q4_K_M"),
    CatalogEntry("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "TinyLlama 1.1B Chat", ModelFamily.LLAMA, "1.1B", ModelTask.CHAT, "Q4_K_M"),
    CatalogEntry("openai/gpt-oss-20b", "GPT-OSS 20B", ModelFamily.OTHER, "20B", ModelTask.CHAT, "Q4_K_M"),
    CatalogEntry("openai/gpt-oss-120b", "GPT-OSS 120B", ModelFamily.OTHER, "120B", ModelTask.CHAT, "Q4_K_M"),
    # Embeddings
    CatalogEntry("BAAI/bge-small-en-v1.5", "BGE Small EN", ModelFamily.OTHER, "33M", ModelTask.EMBEDDING, "F16"),
    CatalogEntry("BAAI/bge-base-en-v1.5", "BGE Base EN", ModelFamily.OTHER, "109M", ModelTask.EMBEDDING, "F16"),
    CatalogEntry("sentence-transformers/all-MiniLM-L6-v2", "MiniLM L6 v2", ModelFamily.OTHER, "22M", ModelTask.EMBEDDING, "F16"),
)


def _parse_param_size(params: str) -> float:
    """Convert param label (e.g. 7B, 22M) to billions of parameters."""
    p = params.strip().upper()
    if p.endswith("B"):
        return float(p[:-1])
    if p.endswith("M"):
        return float(p[:-1]) / 1000
    return float(p)


def search_catalog(
    query: str = "",
    family: str | None = None,
    task: str | None = None,
    max_params: str | None = None,
) -> list[dict]:
    """Search and filter catalog entries."""
    q = query.lower().strip()
    results: list[dict] = []
    for e in CATALOG:
        if family and e.family.value != family.lower():
            continue
        if task and e.task.value != task.lower():
            continue
        if max_params and _parse_param_size(e.params) > _parse_param_size(max_params):
            continue
        if q:
            haystack = f"{e.name} {e.repo_id} {' '.join(e.tags)} {e.family.value}".lower()
            tokens = q.split()
            if not all(tok in haystack for tok in tokens):
                continue
        results.append(
            {
                "repo_id": e.repo_id,
                "name": e.name,
                "family": e.family.value,
                "params": e.params,
                "task": e.task.value,
                "quant": e.quant,
                "tags": list(e.tags),
                "gguf_repo": e.gguf_repo,
            }
        )
    return results


def get_families() -> list[str]:
    return sorted({e.family.value for e in CATALOG})


def get_by_repo(repo_id: str) -> CatalogEntry | None:
    for e in CATALOG:
        if e.repo_id == repo_id:
            return e
    return None
