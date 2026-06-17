"""Tests for inference backend selection and model options."""

from __future__ import annotations

from pathlib import Path

import pytest

from forge.services.inference_models import resolve_chat_target
from seiso.inference.backends import (
    BACKEND_LLAMACPP,
    BACKEND_OLLAMA,
    BACKEND_TORCH,
    available_backends,
    match_ollama_name,
    recommend_backend,
    resolve_gguf_file,
    resolve_local_backend,
)


def test_gguf_recommends_llamacpp(tmp_path: Path):
    gguf = tmp_path / "model-q4.gguf"
    gguf.write_bytes(b"gguf")
    assert recommend_backend(model_path=str(gguf), model_format="gguf") == BACKEND_LLAMACPP


def test_safetensors_recommends_torch_or_mlx(tmp_path: Path):
    model_dir = tmp_path / "merged"
    model_dir.mkdir()
    (model_dir / "model.safetensors").write_bytes(b"x")
    backend = recommend_backend(model_path=str(model_dir), model_format="safetensors")
    assert backend in {BACKEND_TORCH, "mlx"}


def test_available_backends_includes_ollama_when_tag_matches(tmp_path: Path):
    gguf = tmp_path / "my-lora.gguf"
    gguf.write_bytes(b"gguf")
    backends = available_backends(
        model_path=str(gguf),
        model_format="gguf",
        ollama_names={"my-lora:latest", "other"},
    )
    assert BACKEND_LLAMACPP in backends
    assert BACKEND_OLLAMA in backends


def test_match_ollama_name():
    assert match_ollama_name(
        model_path="/models/foo.gguf",
        model_name="foo",
        ollama_names={"foo:latest"},
    ) == "foo:latest"


def test_resolve_gguf_file_picks_largest(tmp_path: Path):
    small = tmp_path / "small.gguf"
    large = tmp_path / "large.gguf"
    small.write_bytes(b"a")
    large.write_bytes(b"a" * 10)
    assert resolve_gguf_file(str(tmp_path)).name == "large.gguf"


def test_resolve_local_backend_auto():
    assert (
        resolve_local_backend(
            model_path="/tmp/model.gguf",
            model_format="gguf",
            requested="auto",
        )
        == BACKEND_LLAMACPP
    )


def test_resolve_chat_target_ollama_kind():
    option = {
        "id": "ollama:llama3.2",
        "kind": "ollama",
        "name": "llama3.2",
        "ollama_model": "llama3.2",
        "default_backend": BACKEND_OLLAMA,
        "backends": [BACKEND_OLLAMA],
        "path": None,
        "format": None,
    }
    target = resolve_chat_target(option, model_id=None, ollama_model=None, inference_backend="auto")
    assert target["inference_backend"] == BACKEND_OLLAMA
    assert target["ollama_model"] == "llama3.2"


def test_resolve_chat_target_local_gguf_ollama_engine():
    option = {
        "id": "local-1",
        "kind": "local",
        "name": "export-q4",
        "path": "/data/export-q4/model.gguf",
        "format": "gguf",
        "default_backend": BACKEND_LLAMACPP,
        "backends": [BACKEND_LLAMACPP, BACKEND_OLLAMA],
        "ollama_model": "export-q4:latest",
    }
    target = resolve_chat_target(option, model_id="local-1", ollama_model=None, inference_backend="ollama")
    assert target["inference_backend"] == BACKEND_OLLAMA
    assert target["ollama_model"] == "export-q4:latest"


def test_resolve_chat_target_ollama_without_tag_raises():
    option = {
        "id": "local-2",
        "kind": "local",
        "name": "orphan",
        "path": "/data/orphan.gguf",
        "format": "gguf",
        "default_backend": BACKEND_LLAMACPP,
        "backends": [BACKEND_LLAMACPP],
        "ollama_model": None,
    }
    with pytest.raises(ValueError, match="not available in Ollama"):
        resolve_chat_target(option, model_id="local-2", ollama_model=None, inference_backend="ollama")


@pytest.mark.asyncio
async def test_local_inference_stream_propagates_errors(monkeypatch):
    from seiso.inference.runner import LocalInferenceRunner

    async def _noop_switch(_path: str) -> None:
        return None

    runner = LocalInferenceRunner()
    monkeypatch.setattr(runner, "_ensure_model_switch", _noop_switch)
    monkeypatch.setattr(runner, "_resolve_route", lambda _payload, _path: ("llama", "/tmp/fake.gguf"))
    monkeypatch.setattr(runner._pool, "bump_generation", lambda: 1)
    monkeypatch.setattr(runner._pool, "is_generation_active", lambda _gen: True)

    def _boom(*_args, **_kwargs):
        raise RuntimeError("model load failed")

    monkeypatch.setattr(runner, "_iter_tokens", _boom)

    with pytest.raises(RuntimeError, match="model load failed"):
        async for _token in runner.stream({"model_path": "/tmp/fake.gguf"}):
            pass
