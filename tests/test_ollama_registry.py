"""Tests for Ollama model registration helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_default_ollama_tag_from_repo_id():
    from forge.services.ollama_registry import default_ollama_tag

    tag = default_ollama_tag("/tmp/model.gguf", repo_id="unsloth/Qwen3-8B-GGUF")
    assert tag == "seiso-unsloth-qwen3-8b-gguf-model"


def test_default_ollama_tag_includes_quant_from_gguf_file():
    from forge.services.ollama_registry import default_ollama_tag

    tag = default_ollama_tag(
        "/tmp/inventory",
        repo_id="org/model",
        metadata={"gguf_file": "model-Q4_K_M.gguf"},
    )
    assert tag == "seiso-org-model-model-q4-k-m"


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

    monkeypatch.setattr(ollama_registry, "load_registry_entries", lambda: {})
    gguf = tmp_path / "model.gguf"
    tag = ollama_registry.resolve_ollama_tag(
        str(gguf),
        metadata={"ollama_tag": "custom-tag"},
    )
    assert tag == "custom-tag"


def test_register_gguf_persists_when_ollama_unavailable(tmp_path, monkeypatch):
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
    assert tag == "seiso-org-model-model"
    saved = json.loads(registry_file.read_text(encoding="utf-8"))
    entry = saved[str(gguf.resolve())]
    assert entry["tag"] == "seiso-org-model-model"
    assert entry["create_skipped"] is True


def test_register_gguf_with_ollama_runs_create(tmp_path, monkeypatch):
    from forge.services import ollama_registry

    registry_file = tmp_path / "ollama_registry.json"
    monkeypatch.setattr(ollama_registry, "registry_path", lambda: registry_file)
    monkeypatch.setattr(ollama_registry, "_ollama_available", lambda: True)
    monkeypatch.setattr(ollama_registry, "_ollama_model_exists", lambda _tag: False)

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
    assert tag == "seiso-org-model-model"
    assert calls and calls[0][:3] == ["ollama", "create", "seiso-org-model-model"]
    saved = json.loads(registry_file.read_text(encoding="utf-8"))
    assert saved[str(gguf.resolve())]["tag"] == "seiso-org-model-model"


def test_register_reference_tag_for_safetensors(tmp_path, monkeypatch):
    from forge.services import ollama_registry

    registry_file = tmp_path / "ollama_registry.json"
    monkeypatch.setattr(ollama_registry, "registry_path", lambda: registry_file)
    monkeypatch.setattr(ollama_registry, "_ollama_available", lambda: False)

    model_dir = tmp_path / "snapshot"
    model_dir.mkdir()
    (model_dir / "model.safetensors").write_bytes(b"weights")

    tag = ollama_registry.register_model_with_ollama(
        str(model_dir),
        repo_id="org/model",
        metadata={"ollama_tag": "llama3.2"},
        model_format="safetensors",
    )
    assert tag == "llama3.2"
    saved = json.loads(registry_file.read_text(encoding="utf-8"))
    assert saved[str(model_dir.resolve())]["tag"] == "llama3.2"


def test_ollama_subprocess_sets_cli_host(monkeypatch):
    from seiso.inference import ollama_registry

    monkeypatch.setenv("SEISO_OLLAMA_URL", "http://127.0.0.1:11434")
    env = ollama_registry._ollama_subprocess_env()
    assert env["OLLAMA_HOST"] == "127.0.0.1:11434"


def test_register_gguf_directory(tmp_path, monkeypatch):
    from forge.services import ollama_registry

    registry_file = tmp_path / "ollama_registry.json"
    monkeypatch.setattr(ollama_registry, "registry_path", lambda: registry_file)
    monkeypatch.setattr(ollama_registry, "_ollama_available", lambda: True)
    monkeypatch.setattr(ollama_registry, "_ollama_model_exists", lambda _tag: False)

    model_dir = tmp_path / "inventory"
    model_dir.mkdir()
    (model_dir / "weights-Q4_K_M.gguf").write_bytes(b"gguf")

    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        import subprocess

        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(ollama_registry.subprocess, "run", fake_run)

    tag = ollama_registry.register_model_with_ollama(
        str(model_dir),
        repo_id="org/model",
        metadata={"gguf_file": "weights-Q4_K_M.gguf"},
        model_format="gguf",
    )
    assert tag == "seiso-org-model-weights-q4-k-m"
    assert calls and calls[0][0] == "ollama"


def test_register_gguf_retries_after_failed_create(tmp_path, monkeypatch):
    from forge.services import ollama_registry

    registry_file = tmp_path / "ollama_registry.json"
    monkeypatch.setattr(ollama_registry, "registry_path", lambda: registry_file)
    monkeypatch.setattr(ollama_registry, "_ollama_available", lambda: True)
    monkeypatch.setattr(ollama_registry, "_ollama_model_exists", lambda _tag: False)

    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        import subprocess

        if len(calls) == 1:
            raise subprocess.CalledProcessError(1, cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(ollama_registry.subprocess, "run", fake_run)

    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"gguf")

    tag = ollama_registry.register_gguf_with_ollama(str(gguf), repo_id="org/model")
    assert tag == "seiso-org-model-model"
    saved = json.loads(registry_file.read_text(encoding="utf-8"))
    assert saved[str(gguf.resolve())]["create_skipped"] is True

    tag = ollama_registry.ensure_model_registered(str(gguf), model_format="gguf")
    assert tag == "seiso-org-model-model"
    assert len(calls) == 2
    saved = json.loads(registry_file.read_text(encoding="utf-8"))
    assert saved[str(gguf.resolve())]["create_skipped"] is False


def test_ensure_model_registered_uses_payload_metadata(tmp_path, monkeypatch):
    from forge.services import ollama_registry

    registry_file = tmp_path / "ollama_registry.json"
    monkeypatch.setattr(ollama_registry, "registry_path", lambda: registry_file)
    monkeypatch.setattr(ollama_registry, "_ollama_available", lambda: False)

    model_dir = tmp_path / "snapshot"
    model_dir.mkdir()
    tag = ollama_registry.ensure_model_registered(
        str(model_dir),
        metadata={"ollama_tag": "existing-model"},
        model_format="safetensors",
    )
    assert tag == "existing-model"
