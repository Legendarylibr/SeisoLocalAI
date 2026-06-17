"""Tests for Hugging Face Hub download helpers."""


from forge.services.hf_hub import _pick_gguf_file
from seiso.models.hf_env import resolve_hf_cache_dir


def test_pick_gguf_prefers_quant():
    files = ["model-Q8_0.gguf", "model-Q4_K_M.gguf", "model-Q5_K_M.gguf"]
    assert _pick_gguf_file(files, preferred_quant="Q4_K_M") == "model-Q4_K_M.gguf"


def test_pick_gguf_fallback():
    files = ["tiny.gguf", "big-model.gguf"]
    assert _pick_gguf_file(files) in files


def test_resolve_hf_cache_env(monkeypatch, tmp_path):
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    assert resolve_hf_cache_dir(tmp_path) == tmp_path / "hf_cache"

    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    assert resolve_hf_cache_dir(tmp_path) == tmp_path / "hf" / "hub"
