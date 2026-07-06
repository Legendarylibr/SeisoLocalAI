"""Tests for llama.cpp vision/mmproj helpers."""

from __future__ import annotations

from seiso.inference.llama_vision import (
    apply_llama_vision_load_kwargs,
    resolve_mmproj_path,
)


def test_resolve_mmproj_path_picks_colocated_file(tmp_path):
    model = tmp_path / "gemma-vision-Q4_K_M.gguf"
    mmproj = tmp_path / "mmproj-Q8_0.gguf"
    model.write_bytes(b"model")
    mmproj.write_bytes(b"mmproj")

    assert resolve_mmproj_path(model) == str(mmproj.resolve())
    assert resolve_mmproj_path(tmp_path / "missing.gguf") is None


def test_resolve_mmproj_path_prefers_matching_quant(tmp_path):
    model = tmp_path / "gemma-vision-Q4_K_M.gguf"
    other = tmp_path / "mmproj-F16.gguf"
    match = tmp_path / "mmproj-Q4_K_M.gguf"
    model.write_bytes(b"model")
    other.write_bytes(b"other")
    match.write_bytes(b"match")

    assert resolve_mmproj_path(model) == str(match.resolve())


def test_repo_likely_needs_mmproj_detects_vision_tags():
    from seiso.inference.llama_vision import repo_likely_needs_mmproj

    assert repo_likely_needs_mmproj(
        "org/Llama-8B",
        tags=("gguf", "vision"),
    )
    assert not repo_likely_needs_mmproj(
        "org/Llama-8B",
        tags=("gguf",),
        gguf_filename="Llama-Q4_K_M.gguf",
    )
    assert repo_likely_needs_mmproj(
        "org/Qwen2.5-VL-7B",
        gguf_filename="Qwen2.5-VL-Q4_K_M.gguf",
    )


def test_apply_llama_vision_load_kwargs_without_mmproj(tmp_path):
    model = tmp_path / "text-only.gguf"
    model.write_bytes(b"model")

    kwargs = apply_llama_vision_load_kwargs({"n_ctx": 2048}, str(model))

    assert "chat_handler" not in kwargs


def test_apply_llama_vision_load_kwargs_attaches_handler(monkeypatch, tmp_path):
    model = tmp_path / "gemma-vision-Q4_K_M.gguf"
    mmproj = tmp_path / "mmproj-Q8_0.gguf"
    model.write_bytes(b"model")
    mmproj.write_bytes(b"mmproj")

    class FakeHandler:
        def __init__(self, *, clip_model_path: str, verbose: bool = False) -> None:
            self.clip_model_path = clip_model_path
            self.verbose = verbose

    monkeypatch.setattr(
        "seiso.inference.llama_vision.build_llama_vision_chat_handler",
        lambda _model, mmproj_path: FakeHandler(clip_model_path=mmproj_path),
    )

    kwargs = apply_llama_vision_load_kwargs({"n_ctx": 2048}, str(model))

    assert isinstance(kwargs["chat_handler"], FakeHandler)
    assert kwargs["chat_handler"].clip_model_path == str(mmproj.resolve())
