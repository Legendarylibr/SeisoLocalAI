"""Tests for HF model catalog."""

from seiso.models.catalog import CATALOG, diversify_by_family, get_by_repo, get_families, search_catalog


def test_catalog_has_recent_qwen_models():
    all_models = search_catalog()
    assert len(all_models) >= 20
    repos = {m["repo_id"] for m in all_models}
    assert "Qwen/Qwen3.6-35B-A3B" in repos
    assert "Qwen/Qwen3.6-27B" in repos
    assert "Qwen/Qwen3.6-4B" in repos
    assert "Qwen/Qwen3.5-4B" in repos
    assert "Qwen/Qwen3.5-2B" in repos
    assert "Qwen/Qwen3-Coder-Next" in repos
    assert "Qwen/Qwen3-8B" not in repos
    assert "Qwen/Qwen2.5-7B-Instruct" not in repos
    assert "Qwen/QwQ-32B" not in repos
    assert "meta-llama/Llama-3.3-70B-Instruct" not in repos
    assert "meta-llama/Llama-3.2-3B-Instruct" not in repos
    assert "mistralai/Mistral-7B-Instruct-v0.3" not in repos
    assert "moonshotai/Kimi-K2.7-Code" in repos
    assert "moonshotai/Kimi-K2-Instruct" in repos
    assert "zai-org/GLM-4.5-Air" in repos
    assert "Qwen/Qwen3-VL-2B-Instruct" in repos
    assert "google/gemma-3-27b-it" in repos
    assert "mistralai/Devstral-Small-2507" in repos
    assert "mistralai/Mistral-Small-3.2-24B-Instruct-2506" in repos
    assert "deepseek-ai/DeepSeek-R1-0528" in repos
    assert "openai/gpt-oss-20b" in repos
    assert "meta-llama/Llama-4-Scout-17B-16E-Instruct" in repos


def test_catalog_covers_major_brands():
    families = set(get_families())
    for brand in ("llama", "qwen", "gemma", "mistral", "deepseek", "phi", "kimi", "glm"):
        assert brand in families


def test_catalog_search():
    results = search_catalog("qwen coder")
    assert any("Coder" in m["name"] for m in results)


def test_catalog_filter_family():
    results = search_catalog(family="gemma")
    assert all(m["family"] == "gemma" for m in results)


def test_get_by_repo():
    entry = get_by_repo("Qwen/Qwen3.6-4B")
    assert entry is not None
    assert entry.family.value == "qwen"


def test_catalog_priority_order():
    results = search_catalog()
    assert results[0]["priority"] >= results[-1]["priority"]
    assert results[0]["repo_id"] == "Qwen/Qwen3.6-35B-A3B"


def test_catalog_search_ranks_exact_match():
    results = search_catalog("qwen 3.6")
    assert results[0]["repo_id"].startswith("Qwen/")
    assert "3.6" in results[0]["name"] or "3.6" in results[0]["repo_id"]


def test_catalog_chat_includes_qwen_36_line():
    results = search_catalog(task="chat")
    repos = {m["repo_id"] for m in results}
    assert "Qwen/Qwen3.6-27B" in repos
    assert "Qwen/Qwen3.5-4B" in repos
    assert "meta-llama/Llama-4-Scout-17B-16E-Instruct" in repos


def test_diversify_by_family_interleaves_brands():
    models = search_catalog()
    diversified = diversify_by_family(models[:24])
    first_families = [m["family"] for m in diversified[:12]]
    assert len(set(first_families)) >= 5


def test_catalog_qwen_36_dominates_top_entries():
    qwen36 = [e for e in CATALOG if "3.6" in e.repo_id]
    assert len(qwen36) >= 4
    assert max(e.priority for e in qwen36) == 100
