"""Tests for Hugging Face Hub download helpers."""

import pytest

from forge.services.hf_hub import (
    _complete_shard_group_for,
    _pick_gguf_file,
    _pick_gguf_files,
)
from seiso.models.hf_env import resolve_hf_cache_dir


def test_pick_gguf_prefers_quant():
    files = ["model-Q8_0.gguf", "model-Q4_K_M.gguf", "model-Q5_K_M.gguf"]
    assert _pick_gguf_file(files, preferred_quant="Q4_K_M") == "model-Q4_K_M.gguf"


def test_list_complete_gguf_file_groups_returns_all_quants():
    from forge.services.hf_hub_gguf_select import list_complete_gguf_file_groups

    files = [
        "model-Q5_K_M.gguf",
        "model-Q4_K_M.gguf",
        "mmproj-f16.gguf",
        "model-Q8_0.gguf",
    ]
    groups = list_complete_gguf_file_groups(files)
    flat = {g[0] for g in groups}
    assert flat == {
        "model-Q4_K_M.gguf",
        "model-Q5_K_M.gguf",
        "model-Q8_0.gguf",
    }
    assert all(len(g) == 1 for g in groups)


def test_pick_gguf_prefers_novel_quant():
    files = ["model-Q8_0.gguf", "model-Q4_K_XL.gguf", "model-Q5_K_M.gguf"]
    assert _pick_gguf_file(files, preferred_quant="Q4_K_XL") == "model-Q4_K_XL.gguf"


def test_pick_gguf_fallback():
    files = ["tiny.gguf", "big-model.gguf"]
    assert _pick_gguf_file(files) in files


def test_pick_gguf_files_returns_complete_shard_group():
    files = [
        "model-Q4_K_M-00002-of-00002.gguf",
        "model-Q4_K_M-00001-of-00002.gguf",
        "model-Q8_0.gguf",
    ]

    assert _pick_gguf_files(files, preferred_quant="Q4_K_M") == [
        "model-Q4_K_M-00001-of-00002.gguf",
        "model-Q4_K_M-00002-of-00002.gguf",
    ]


def test_pick_gguf_files_rejects_incomplete_shard_group():
    files = ["model-Q4_K_M-00001-of-00002.gguf"]

    assert _pick_gguf_files(files, preferred_quant="Q4_K_M") == []


def test_pick_gguf_files_rejects_duplicate_shard_indices():
    files = [
        "model-Q4_K_M-00001-of-00003.gguf",
        "model-Q4_K_M-00001-of-00003.gguf",
        "model-Q4_K_M-00002-of-00003.gguf",
    ]

    assert _pick_gguf_files(files, preferred_quant="Q4_K_M") == []


def test_pick_gguf_files_rejects_incomplete_even_with_non_sharded():
    files = [
        "model-Q4_K_M-00001-of-00002.gguf",
        "other-Q4_K_M.gguf",
    ]

    assert _pick_gguf_files(files, preferred_quant="Q4_K_M") == []


def test_complete_shard_group_for_rejects_incomplete_explicit_filename():
    files = ["model-Q4_K_M-00001-of-00002.gguf"]
    with pytest.raises(ValueError, match="Incomplete GGUF shard"):
        _complete_shard_group_for(files, "model-Q4_K_M-00001-of-00002.gguf")


def test_complete_shard_group_for_returns_full_group():
    files = [
        "model-Q4_K_M-00002-of-00002.gguf",
        "model-Q4_K_M-00001-of-00002.gguf",
    ]
    assert _complete_shard_group_for(files, "model-Q4_K_M-00001-of-00002.gguf") == [
        "model-Q4_K_M-00001-of-00002.gguf",
        "model-Q4_K_M-00002-of-00002.gguf",
    ]


def test_resolve_hf_cache_env(monkeypatch, tmp_path):
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    assert resolve_hf_cache_dir(tmp_path) == tmp_path / "hf_cache"

    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    assert resolve_hf_cache_dir(tmp_path) == tmp_path / "hf" / "hub"
