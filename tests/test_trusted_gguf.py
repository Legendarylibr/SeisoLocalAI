"""Tests for GGUF repo filtering and ranking."""

from seiso.models.trusted_gguf import (
    filter_trusted_gguf_search_results,
    gguf_mirror_candidates,
    is_trusted_gguf_repo,
    rank_trusted_gguf_repos,
)


def test_any_hf_repo_is_supported():
    assert is_trusted_gguf_repo("unsloth/Qwen3.6-4B-GGUF")
    assert is_trusted_gguf_repo("random-user/base-model")
    assert is_trusted_gguf_repo("Qwen/Qwen2.5-0.5B-Instruct")
    assert is_trusted_gguf_repo("vendor/Kimi-DFlash")


def test_gguf_mirror_candidates_probe_naming_variants():
    candidates = gguf_mirror_candidates("acme/Example-7B")
    assert candidates == ["acme/Example-7B-GGUF", "acme/Example-7B"]


def test_filter_trusted_gguf_search_results_sorts_by_downloads():
    rows = [
        {"repo_id": "random-user/Llama-GGUF", "downloads": 999_999},
        {"repo_id": "bartowski/Llama-GGUF", "downloads": 100},
        {"repo_id": "meta-llama/Llama-GGUF", "downloads": 50},
    ]
    filtered = filter_trusted_gguf_search_results(rows, base_repo_id="meta-llama/Llama-3.1-8B")
    assert [row["repo_id"] for row in filtered] == [
        "random-user/Llama-GGUF",
        "bartowski/Llama-GGUF",
        "meta-llama/Llama-GGUF",
    ]


def test_rank_trusted_gguf_repos_sorts_by_downloads():
    ordered = rank_trusted_gguf_repos(
        ["bartowski/A-GGUF", "QuantFactory/A-GGUF"],
        popularity={"QuantFactory/A-GGUF": 200, "bartowski/A-GGUF": 50},
    )
    assert ordered == ["QuantFactory/A-GGUF", "bartowski/A-GGUF"]