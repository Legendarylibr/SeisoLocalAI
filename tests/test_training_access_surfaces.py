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
from seiso.research.nostr.keys import generate_keypair
from seiso.training.access import (
    FRONTEND_SURFACE,
    assert_surface_distributed_config,
    frontend_training_surface,
)


@pytest.fixture
def valid_buzz_nsec() -> str:
    return generate_keypair().nsec


def test_frontend_surface_exposes_full_training_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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


def test_desktop_local_training_needs_neither_buzz_nor_mesh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Desktop Forge / plain ``seiso train`` must work with Buzz+mesh completely unset.

    Mesh is only for Buzz-room agent multi-node — never a prerequisite for local runs.
    """
    for key in (
        "SEISO_ALLOW_MESH",
        "SEISO_MESH_TOKEN",
        "SEISO_MESH_ALLOW_LOOPBACK",
        "BUZZ_PRIVATE_KEY",
        "BUZZ_AUTH_TAG",
        "SEISO_AGENT",
        "SEISO_TRAINING_SURFACE",
    ):
        monkeypatch.delenv(key, raising=False)

    assert buzz_agent_present() is False
    from seiso.mesh.flags import mesh_allowed, require_mesh_allowed

    assert mesh_allowed() is False
    with pytest.raises(RuntimeError, match="SEISO_ALLOW_MESH"):
        require_mesh_allowed()

    surface = frontend_training_surface()
    assert surface["mesh"]["available_on_this_surface"] is False
    assert surface["mesh"]["buzz_agent_present"] is False
    assert resolve_training_surface() == TrainingSurface.FRONTEND

    # Local single-node (including multi-GPU DDP nnodes=1) stays open.
    assert_surface_distributed_config(
        FRONTEND_SURFACE,
        {"distributed_num_nodes": 1, "multi_gpu": True},
    )

    from seiso.training.config import TrainConfig
    from seiso.training.multi_gpu import detect_training_layout, resolve_distributed_plan

    cfg = TrainConfig.from_yaml(Path("configs/smoke_train_cpu.yaml"))
    assert int(cfg.distributed_num_nodes) == 1
    plan = resolve_distributed_plan(cfg, detect_training_layout())
    assert plan.nnodes == 1

    # Mesh CLI helpers stay fail-closed — desktop never needs them.
    from seiso.mesh.coordinator import announce, build_plan

    with pytest.raises(RuntimeError, match="SEISO_ALLOW_MESH"):
        announce(channel="desktop", gpus=1)
    with pytest.raises(RuntimeError, match="SEISO_ALLOW_MESH"):
        build_plan(
            channel="desktop",
            job_type="finetune",
            nodes=2,
            master_addr="10.0.0.1",
            gpus_per_node=1,
        )


def test_frontend_refuses_multinode_config() -> None:
    with pytest.raises(ValueError, match="Buzz-agent/mesh-only"):
        assert_surface_distributed_config(
            FRONTEND_SURFACE,
            {"distributed_num_nodes": 2},
        )
    with pytest.raises(ValueError, match="Buzz-agent/mesh-only|nemo_rl"):
        assert_surface_distributed_config(
            FRONTEND_SURFACE,
            {"distributed_num_nodes": 1, "nemo_rl_num_nodes": 2},
        )
    assert_surface_distributed_config(
        FRONTEND_SURFACE,
        {"distributed_num_nodes": 1, "multi_gpu": True, "nemo_rl_num_nodes": 1},
    )


def test_agent_surface_allows_multinode_only_with_mesh_and_buzz(
    monkeypatch: pytest.MonkeyPatch,
    valid_buzz_nsec: str,
) -> None:
    monkeypatch.delenv("SEISO_ALLOW_MESH", raising=False)
    monkeypatch.delenv("BUZZ_PRIVATE_KEY", raising=False)
    with pytest.raises(ValueError, match="Buzz-agent/mesh-only|BUZZ_PRIVATE_KEY|Multi-node"):
        assert_surface_distributed_config(
            TrainingSurface.AGENT,
            {"distributed_num_nodes": 4},
        )
    monkeypatch.setenv("SEISO_ALLOW_MESH", "1")
    monkeypatch.setenv("BUZZ_PRIVATE_KEY", valid_buzz_nsec)
    assert_surface_distributed_config(
        TrainingSurface.AGENT,
        {"distributed_num_nodes": 4},
    )


def test_buzz_agent_required_for_mesh_feature(
    monkeypatch: pytest.MonkeyPatch,
    valid_buzz_nsec: str,
) -> None:
    monkeypatch.delenv("BUZZ_PRIVATE_KEY", raising=False)
    monkeypatch.delenv("BUZZ_AUTH_TAG", raising=False)
    assert buzz_agent_present() is False
    with pytest.raises(RuntimeError, match="Buzz-agent-only"):
        require_buzz_agent(feature="Mesh")
    monkeypatch.setenv("BUZZ_PRIVATE_KEY", "nsec1test-not-a-real-key")
    assert buzz_agent_present() is False
    with pytest.raises(RuntimeError, match="valid Buzz agent nsec"):
        require_buzz_agent(feature="Mesh")
    monkeypatch.setenv("BUZZ_PRIVATE_KEY", valid_buzz_nsec)
    assert buzz_agent_present() is True
    require_buzz_agent(feature="Mesh")


def test_trivial_buzz_auth_tag_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BUZZ_PRIVATE_KEY", raising=False)
    monkeypatch.setenv("BUZZ_AUTH_TAG", "1")
    assert buzz_agent_present() is False
    with pytest.raises(RuntimeError, match="non-trivial BUZZ_AUTH_TAG"):
        require_buzz_agent(feature="Mesh")
    monkeypatch.setenv("BUZZ_AUTH_TAG", "desktop-managed-session-tag")
    assert buzz_agent_present() is True
    require_buzz_agent(feature="Mesh")


def test_resolve_training_surface_defaults_frontend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    monkeypatch.setenv("SEISO_MESH_TOKEN", "shared-secret-16+")
    monkeypatch.delenv("BUZZ_PRIVATE_KEY", raising=False)
    monkeypatch.delenv("BUZZ_AUTH_TAG", raising=False)
    from seiso.mesh.flags import require_mesh_allowed

    with pytest.raises(RuntimeError, match="BUZZ_PRIVATE_KEY"):
        require_mesh_allowed()


def test_mesh_auth_tag_alone_cannot_sign(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SEISO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SEISO_ALLOW_MESH", "1")
    monkeypatch.setenv("SEISO_MESH_TOKEN", "shared-secret-16+")
    monkeypatch.delenv("BUZZ_PRIVATE_KEY", raising=False)
    monkeypatch.setenv("BUZZ_AUTH_TAG", "desktop-managed-session-tag")
    from seiso.mesh.flags import require_mesh_allowed

    with pytest.raises(RuntimeError, match="BUZZ_PRIVATE_KEY"):
        require_mesh_allowed()


def test_agent_receipt_scrubs_secrets() -> None:
    from seiso.agent.receipts import agent_receipt, channel_safe_plan_view

    receipt = agent_receipt(
        role="plan",
        status="planned",
        job_id="abc",
        token="leak",
        token_fingerprint="deadbeef",
        nsec="nsec1leak",
        mesh_token="secret",
        hostname="devs-MacBook-Pro.local",
        host="localbox",
    )
    assert "token" not in receipt
    assert "token_fingerprint" not in receipt
    assert "nsec" not in receipt
    assert "mesh_token" not in receipt
    assert "hostname" not in receipt
    assert "host" not in receipt
    assert receipt["job_id"] == "abc"
    view = channel_safe_plan_view(
        {
            "job_id": "abc",
            "token_fingerprint": "deadbeef",
            "distributed_num_nodes": 2,
            "hostname": "should-not-appear",
        }
    )
    assert "token_fingerprint" not in view
    assert "hostname" not in view
    assert view["distributed_num_nodes"] == 2
