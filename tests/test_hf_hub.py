"""Tests for Hugging Face Hub download helpers."""

from forge.services.hf_hub import _pick_gguf_file, _pick_gguf_files
from seiso.models.hf_env import resolve_hf_cache_dir


def test_pick_gguf_prefers_quant():
    files = ["model-Q8_0.gguf", "model-Q4_K_M.gguf", "model-Q5_K_M.gguf"]
    assert _pick_gguf_file(files, preferred_quant="Q4_K_M") == "model-Q4_K_M.gguf"


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


def test_resolve_hf_cache_env(monkeypatch, tmp_path):
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    assert resolve_hf_cache_dir(tmp_path) == tmp_path / "hf_cache"

    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    assert resolve_hf_cache_dir(tmp_path) == tmp_path / "hf" / "hub"
