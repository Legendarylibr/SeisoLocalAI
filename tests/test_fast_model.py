"""Tests for FastModel API."""

from seiso.models.fast_model import resolve_dtype
from seiso.export.gguf import write_ollama_modelfile


def test_resolve_dtype_default():
    assert resolve_dtype(None) in (None, "bfloat16", "float16")


def test_ollama_modelfile():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        mf = write_ollama_modelfile(d, "model-q4_k_m.gguf", model_name="test-model")
        text = mf.read_text()
        assert "FROM ./model-q4_k_m.gguf" in text
        assert "PARAMETER temperature" in text
