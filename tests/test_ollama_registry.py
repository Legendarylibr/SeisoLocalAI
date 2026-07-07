"""Tests for Ollama GGUF registration helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_default_ollama_tag_from_repo_id():
    from forge.services.ollama_registry import default_ollama_tag

    tag = default_ollama_tag("/tmp/model.gguf", repo_id="unsloth/Qwen3-8B-GGUF")
    assert tag == "seiso-unsloth-qwen3-8b-gguf"


def test_registry_roundtrip(tmp_path, monkeypatch):
    from forge.services import ollama_registry

    registry_file = tmp_path / "ollama_registry.json"
    monkeypatch.setattr(ollama_registry, "registry_path", lambda: registry_file)

    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"gguf")

    ollama_registry.save_registry({str(gguf.resolve()): "seiso-test-model"})
    loaded = ollama_registry.load_registry()
    assert loaded[str(gguf.resolve())] == "seiso-test-model"


def test_resolve_ollama_tag_uses_metadata(monkeypatch, tmp_path):
    from forge.services import ollama_registry

    monkeypatch.setattr(ollama_registry, "load_registry", lambda: {})
    gguf = tmp_path / "model.gguf"
    tag = ollama_registry.resolve_ollama_tag(
        str(gguf),
        metadata={"ollama_tag": "custom-tag"},
    )
    assert tag == "custom-tag"


def test_register_gguf_with_ollama_skips_when_ollama_unavailable(
    tmp_path, monkeypatch
):
    from forge.services import ollama_registry

    registry_file = tmp_path / "ollama_registry.json"
    monkeypatch.setattr(ollama_registry, "registry_path", lambda: registry_file)
    monkeypatch.setattr(ollama_registry, "_ollama_available", lambda: False)

    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"gguf")
    tag = ollama_registry.register_gguf_with_ollama(
        str(gguf),
        repo_id="org/model",
    )
    assert tag == "seiso-org-model"
    assert not registry_file.exists()


def test_register_gguf_with_ollama_runs_create(tmp_path, monkeypatch):
    from forge.services import ollama_registry

    registry_file = tmp_path / "ollama_registry.json"
    monkeypatch.setattr(ollama_registry, "registry_path", lambda: registry_file)
    monkeypatch.setattr(ollama_registry, "_ollama_available", lambda: True)

    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        import subprocess

        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(ollama_registry.subprocess, "run", fake_run)

    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"gguf")
    tag = ollama_registry.register_gguf_with_ollama(
        str(gguf),
        repo_id="org/model",
    )
    assert tag == "seiso-org-model"
    assert calls and calls[0][:3] == ["ollama", "create", "seiso-org-model"]
    saved = json.loads(registry_file.read_text(encoding="utf-8"))
    assert saved[str(gguf.resolve())] == "seiso-org-model"
