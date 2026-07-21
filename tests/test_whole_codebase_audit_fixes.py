"""Regression tests for whole-codebase audit surgical remediations."""

from __future__ import annotations

from pathlib import Path

import pytest

from forge.orchestrators.inference import InferenceOrchestrator
from seiso.compress.runner import _assert_full_model_dir, _resolve_model_dir
from seiso.export.formats import ExportFormat, _select_hub_folder


def test_stale_end_generation_does_not_clear_newer_reservation(tmp_path: Path):
    orch = InferenceOrchestrator(tmp_path)
    epoch1 = orch.begin_generation_for_user("user-a")
    orch.end_generation_for_user("user-a", epoch=epoch1)
    epoch2 = orch.begin_generation_for_user("user-a")
    # Stale finally from the first stream must not clear the new reservation.
    orch.end_generation_for_user("user-a", epoch=epoch1)
    assert orch._active_generation_user_id == "user-a"
    assert orch._active_generation_epoch == epoch2
    orch.end_generation_for_user("user-a", epoch=epoch2)
    assert orch._active_generation_user_id is None


def test_select_hub_folder_skips_empty_gguf_dirs(tmp_path: Path):
    empty = tmp_path / "q4_k_m"
    empty.mkdir()
    good = tmp_path / "q8_0"
    good.mkdir()
    (good / "model-q8_0.gguf").write_bytes(b"gguf")
    chosen = _select_hub_folder(tmp_path, [ExportFormat.GGUF])
    assert chosen == good


def test_compress_refuses_lora_only_model_dir(tmp_path: Path):
    lora = tmp_path / "adapter"
    lora.mkdir()
    (lora / "adapter_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="LoRA adapter only"):
        _assert_full_model_dir(lora, "prune")

    cfg = {
        "model_dir": str(lora),
        "stages": ["prune", "finetune"],
    }
    with pytest.raises(ValueError, match="LoRA adapter only"):
        _resolve_model_dir(cfg, tmp_path / "run", "prune")


def test_csrf_empty_bearer_helper():
    from forge.security.csrf import validate_csrf
    from starlette.requests import Request

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/api/inference/threads",
        "raw_path": b"/api/inference/threads",
        "query_string": b"",
        "headers": [(b"authorization", b"Bearer ")],
        "client": ("127.0.0.1", 123),
        "server": ("test", 80),
    }
    assert validate_csrf(Request(scope)) is False
