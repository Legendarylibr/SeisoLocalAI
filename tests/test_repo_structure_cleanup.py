"""Structural checks for whole-repo dead-code cleanup (shipped entry paths)."""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import pytest


def test_core_packages_import():
    """Documented product packages import without error."""
    import forge
    import seiso
    import seiso_cli

    assert seiso is not None
    assert forge is not None
    assert seiso_cli is not None


def test_dead_process_lock_module_removed():
    """Compat alias module had zero callers — must stay gone."""
    path = Path(__file__).resolve().parents[1] / "seiso" / "inference" / "process_lock.py"
    assert not path.is_file()
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("seiso.inference.process_lock")


def test_removed_dead_helpers_absent():
    """Symbols proven unreferenced must not reappear on the live surface."""
    from seiso.chat import thinking
    from seiso.hardware import tiers
    from seiso.inference import benchmark, llama_vision
    from seiso.memory.protection import device_map
    from seiso.models import moe_sizing, trainable_snapshot, trusted_gguf

    for mod, name in (
        (trusted_gguf, "gguf_repo_owner"),
        (trusted_gguf, "gguf_repo_trust_rank"),
        (moe_sizing, "sizing_for_local_file"),
        (trainable_snapshot, "trainable_weight_files"),
        (llama_vision, "gguf_filename_suggests_vision"),
        (benchmark, "run_benchmark_kv_scenarios"),
        (thinking, "reasoning_quality_system_suffix"),
        (tiers, "ui_headroom_mb"),
        (device_map, "jsonl_load_safe"),
    ):
        assert not hasattr(mod, name), f"{mod.__name__}.{name} should be removed"


def test_ollama_think_max_tokens_single_policy():
    """llamaswap back-compat must call the shared chat.thinking policy (real path)."""
    from seiso.chat.thinking import ollama_think_max_tokens as chat_fn
    from seiso.chat.thinking import thinking_max_tokens
    from seiso.inference.llamaswap import ollama_think_max_tokens as swap_fn

    # Same numeric policy for general task.
    expected = thinking_max_tokens(768, task="general")
    assert chat_fn(768) == expected
    assert swap_fn(768) == expected
    # llamaswap wrapper is not a second implementation: source delegates to chat.
    source = inspect.getsource(swap_fn)
    assert "seiso.chat.thinking" in source
    assert "thinking_max_tokens" in source or "ollama_think_max_tokens" in source


def test_trainable_snapshot_and_trusted_gguf_still_live():
    """Cleanup must not break the helpers callers still use."""
    from seiso.models.trainable_snapshot import snapshot_has_trainable_weights
    from seiso.models.trusted_gguf import (
        is_supported_gguf_repo_candidate,
        is_trusted_gguf_repo,
        rank_trusted_gguf_repos,
    )

    assert is_supported_gguf_repo_candidate("org/model")
    assert is_trusted_gguf_repo("org/model")
    assert rank_trusted_gguf_repos(["b/x", "a/y"]) == ["a/y", "b/x"]
    # Non-existent path: no weight files
    assert snapshot_has_trainable_weights(Path("/nonexistent/seiso-structure-test")) is False


def test_resolve_training_device_map_still_exported():
    from seiso.memory.protection.device_map import resolve_training_device_map

    # Single-process CPU/default path returns None or auto depending on torch.
    result = resolve_training_device_map(device="cpu")
    assert result is None or result == "auto" or isinstance(result, dict)
