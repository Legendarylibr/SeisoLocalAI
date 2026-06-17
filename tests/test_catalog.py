"""Tests for HF model catalog."""

from seiso.models.catalog import diversify_by_family, get_by_repo, get_families, search_catalog


def test_catalog_has_popular_models():
    all_models = search_catalog()
    assert len(all_models) >= 40
    repos = {m["repo_id"] for m in all_models}
    assert "Qwen/Qwen3.6-35B-A3B" in repos
    assert "Qwen/Qwen3.5-4B" in repos
    assert "Qwen/Qwen2.5-7B-Instruct" not in repos
    assert "Qwen/Qwen3-8B" not in repos
    assert "moonshotai/Kimi-K2.7-Code" in repos
    assert "moonshotai/Kimi-K2.5" in repos
    assert "moonshotai/Kimi-K2-Thinking" in repos
    assert "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16" in repos
    assert "zai-org/GLM-5" in repos
    assert "MiniMaxAI/MiniMax-M2.7" in repos
    assert "Qwen/Qwen3-VL-2B-Instruct" in repos
    assert "google/gemma-4-12B-it" in repos
    assert "google/gemma-4-E2B-it" in repos
    assert "MiniMaxAI/MiniMax-M3" in repos
    assert "mistralai/Devstral-Small-2507" in repos
    assert "google/gemma-3-27b-it" not in repos
    assert "mistralai/Mixtral-8x7B-Instruct-v0.1" not in repos
    assert "mistralai/Mistral-7B-Instruct-v0.3" not in repos
    assert "deepseek-ai/DeepSeek-V3" not in repos
    assert "deepseek-ai/DeepSeek-R1" not in repos
    assert "HuggingFaceTB/SmolLM2-1.7B-Instruct" not in repos
    assert "deepseek-ai/deepseek-coder-6.7b-instruct" not in repos
    assert "meta-llama/Llama-4-Scout-17B-16E-Instruct" in repos
    assert "meta-llama/Llama-4-Maverick-17B-128E-Instruct" in repos
    assert "meta-llama/Llama-3.3-70B-Instruct" not in repos
    assert "mistralai/Mistral-Small-4-119B-2603" in repos
    assert "mistralai/Mistral-Small-Instruct-2409" not in repos
    assert "deepseek-ai/DeepSeek-R1-0528" in repos
    assert "ibm-granite/granite-4.0-h-small" in repos
    assert "allenai/Olmo-3-7B-Instruct" in repos


def test_catalog_covers_major_brands():
    families = set(get_families())
    for brand in ("llama", "qwen", "gemma", "mistral", "deepseek", "phi", "kimi", "glm", "ibm", "olmo"):
        assert brand in families


def test_catalog_search():
    results = search_catalog("qwen coder")
    assert any("Coder" in m["name"] for m in results)


def test_catalog_filter_family():
    results = search_catalog(family="gemma")
    assert all(m["family"] == "gemma" for m in results)


def test_get_by_repo():
    entry = get_by_repo("microsoft/phi-4")
    assert entry is not None
    assert entry.family.value == "phi"


def test_catalog_priority_order():
    results = search_catalog()
    assert results[0]["priority"] >= results[-1]["priority"]
    assert results[0]["repo_id"] == "moonshotai/Kimi-K2.7-Code"


def test_catalog_search_ranks_exact_match():
    results = search_catalog("qwen 3.6")
    assert results[0]["repo_id"].startswith("Qwen/")
    assert "3.6" in results[0]["name"] or "3.6" in results[0]["repo_id"]


def test_catalog_chat_includes_qwen_36():
    results = search_catalog(task="chat")
    repos = {m["repo_id"] for m in results}
    assert "Qwen/Qwen3.6-35B-A3B" in repos
    assert "meta-llama/Llama-4-Scout-17B-16E-Instruct" in repos


def test_diversify_by_family_interleaves_brands():
    models = search_catalog()
    diversified = diversify_by_family(models[:24])
    first_families = [m["family"] for m in diversified[:12]]
    assert len(set(first_families)) >= 6
