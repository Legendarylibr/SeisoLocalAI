"""Tests for HF model catalog."""

from seiso.models.catalog import get_families, search_catalog, get_by_repo


def test_catalog_has_popular_models():
    all_models = search_catalog()
    assert len(all_models) >= 40
    repos = {m["repo_id"] for m in all_models}
    assert "meta-llama/Llama-3.2-1B-Instruct" in repos
    assert "Qwen/Qwen2.5-7B-Instruct" in repos


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
