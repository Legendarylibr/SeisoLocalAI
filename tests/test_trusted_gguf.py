"""Tests for trusted GGUF publisher filtering."""

from seiso.models.trusted_gguf import (
    filter_trusted_gguf_search_results,
    gguf_mirror_candidates,
    is_trusted_gguf_repo,
    rank_trusted_gguf_repos,
)


def test_trusted_publishers_include_common_mirrors():
    assert is_trusted_gguf_repo("unsloth/Qwen3.6-4B-GGUF")
    assert is_trusted_gguf_repo("bartowski/Llama-3.1-8B-Instruct-GGUF")
    assert is_trusted_gguf_repo("QuantFactory/Mistral-7B-GGUF")


def test_untrusted_random_gguf_rejected():
    assert not is_trusted_gguf_repo("random-user/my-model-gguf")
    assert not is_trusted_gguf_repo("vendor/Kimi-DFlash")


def test_official_publisher_gguf_allowed():
    assert is_trusted_gguf_repo("Qwen/Qwen3.6-4B-GGUF", base_repo_id="Qwen/Qwen3.6-4B")


def test_catalog_mirror_is_trusted():
    assert is_trusted_gguf_repo("AesSedai/Kimi-K2.7-Code-GGUF")


def test_gguf_mirror_candidates_prioritize_unsloth():
    candidates = gguf_mirror_candidates("meta-llama/Llama-3.1-8B-Instruct")
    assert candidates[0].startswith("unsloth/")
    assert "bartowski/Llama-3.1-8B-Instruct-GGUF" in candidates


def test_filter_trusted_gguf_search_results_ranks_by_trust_and_downloads():
    rows = [
        {"repo_id": "random-user/Llama-GGUF", "downloads": 999_999},
        {"repo_id": "bartowski/Llama-GGUF", "downloads": 100},
        {"repo_id": "unsloth/Llama-GGUF", "downloads": 50},
    ]
    filtered = filter_trusted_gguf_search_results(rows, base_repo_id="meta-llama/Llama-3.1-8B")
    assert [row["repo_id"] for row in filtered] == ["unsloth/Llama-GGUF", "bartowski/Llama-GGUF"]


def test_rank_trusted_gguf_repos_preserves_input_order_for_equal_rank():
    ordered = rank_trusted_gguf_repos(
        ["bartowski/A-GGUF", "QuantFactory/A-GGUF"],
        base_repo_id="org/A",
    )
    assert ordered[0].startswith("bartowski/")
    assert len(ordered) == 2
