"""Frontend vs agent training surfaces + Buzz-agent mesh gate."""

from __future__ import annotations

from pathlib import Path

import pytest

from seiso.agent.surface import (
    TrainingSurface,
    buzz_agent_present,
    require_buzz_agent,
    resolve_training_surface,
)
from seiso.training.access import (
    FRONTEND_SURFACE,
    assert_surface_distributed_config,
    frontend_training_surface,
)


def test_frontend_surface_exposes_full_training_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SEISO_ALLOW_MESH", raising=False)
    monkeypatch.delenv("BUZZ_PRIVATE_KEY", raising=False)
    surface = frontend_training_surface()
    assert surface["surface"] == "frontend"
    assert surface["exposes_full_training_config"] is True
    assert "method" in surface["config_fields"]
    assert "distributed_strategy" in surface["config_fields"]
    assert surface["local_distributed"]["max_nodes"] == 1
    assert surface["multi_node"] is False
    assert surface["mesh"]["available_on_this_surface"] is False


def test_frontend_refuses_multinode_config() -> None:
    with pytest.raises(ValueError, match="Buzz-agent/mesh-only"):
        assert_surface_distributed_config(
            FRONTEND_SURFACE,
            {"distributed_num_nodes": 2},
        )
    assert_surface_distributed_config(
        FRONTEND_SURFACE,
        {"distributed_num_nodes": 1, "multi_gpu": True},
    )


def test_agent_surface_allows_multinode_config() -> None:
    assert_surface_distributed_config(
        TrainingSurface.AGENT,
        {"distributed_num_nodes": 4},
    )


def test_buzz_agent_required_for_mesh_feature(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BUZZ_PRIVATE_KEY", raising=False)
    monkeypatch.delenv("BUZZ_AUTH_TAG", raising=False)
    assert buzz_agent_present() is False
    with pytest.raises(RuntimeError, match="Buzz-agent-only"):
        require_buzz_agent(feature="Mesh")
    monkeypatch.setenv("BUZZ_PRIVATE_KEY", "nsec1test-not-a-real-key")
    assert buzz_agent_present() is True
    require_buzz_agent(feature="Mesh")


def test_resolve_training_surface_defaults_frontend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SEISO_AGENT", raising=False)
    monkeypatch.delenv("SEISO_TRAINING_SURFACE", raising=False)
    monkeypatch.delenv("BUZZ_PRIVATE_KEY", raising=False)
    monkeypatch.delenv("BUZZ_AUTH_TAG", raising=False)
    assert resolve_training_surface() == TrainingSurface.FRONTEND
    monkeypatch.setenv("SEISO_AGENT", "1")
    assert resolve_training_surface() == TrainingSurface.AGENT


def test_mesh_requires_buzz_agent_even_when_allowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SEISO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SEISO_ALLOW_MESH", "1")
    monkeypatch.setenv("SEISO_MESH_TOKEN", "shared")
    monkeypatch.delenv("BUZZ_PRIVATE_KEY", raising=False)
    monkeypatch.delenv("BUZZ_AUTH_TAG", raising=False)
    from seiso.mesh.flags import require_mesh_allowed

    with pytest.raises(RuntimeError, match="Buzz-agent-only"):
        require_mesh_allowed()
