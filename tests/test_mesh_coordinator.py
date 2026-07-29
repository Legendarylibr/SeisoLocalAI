"""Experimental Buzz mesh — opt-in, Buzz-agent-only, no protocol fee."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture()
def mesh_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("SEISO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SEISO_ALLOW_MESH", "1")
    monkeypatch.setenv("SEISO_MESH_TOKEN", "test-mesh-token")
    monkeypatch.setenv("BUZZ_PRIVATE_KEY", "nsec1test-mesh-agent")
    monkeypatch.setenv("SEISO_AGENT", "1")
    return tmp_path / "data"


def test_mesh_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SEISO_ALLOW_MESH", raising=False)
    monkeypatch.setenv("BUZZ_PRIVATE_KEY", "nsec1x")
    from seiso.mesh.flags import mesh_allowed, require_mesh_allowed

    assert mesh_allowed() is False
    with pytest.raises(RuntimeError, match="SEISO_ALLOW_MESH"):
        require_mesh_allowed()


def test_announce_plan_worker(mesh_env: Path) -> None:
    from seiso.mesh.coordinator import (
        announce,
        build_plan,
        load_plan,
        worker_env,
        worker_train_config_overlay,
    )

    ann = announce(channel="ch-1", gpus=2, capabilities=["finetune", "slime"])
    assert ann["buzz_receipt"]["role"] == "announce"
    assert ann["agent_receipt"]["buzz_compatible"] is True
    assert "test-mesh-token" not in json_dumps(ann["buzz_receipt"])
    assert "token" not in ann["buzz_receipt"]

    plan_out = build_plan(
        channel="ch-1",
        job_type="finetune",
        nodes=2,
        master_addr="10.0.0.2",
        gpus_per_node=2,
    )
    assert plan_out["plan"]["protocol_fee_sats"] == 0
    assert plan_out["plan"]["market"] is False
    assert plan_out["plan"]["distributed_nproc_per_node"] == 2
    assert plan_out["plan"]["token_fingerprint"]
    assert plan_out["buzz_receipt"]["world_size"] == 2

    plan = load_plan(plan_out["plan"]["job_id"])
    env = worker_env(plan, node_rank=1)
    assert env["NODE_RANK"] == "1"
    assert env["NNODES"] == "2"
    assert env["MASTER_ADDR"] == "10.0.0.2"
    assert env["NPROC_PER_NODE"] == "2"
    overlay = worker_train_config_overlay(plan, node_rank=1)
    assert overlay["distributed_nproc_per_node"] == 2
    assert overlay["distributed_node_rank"] == 1


def test_build_plan_requires_gpus_per_node(mesh_env: Path) -> None:
    from seiso.mesh.coordinator import build_plan

    with pytest.raises(ValueError, match="gpus_per_node is required"):
        build_plan(
            channel="ch-1",
            job_type="finetune",
            nodes=2,
            master_addr="10.0.0.2",
        )


def test_mesh_token_mismatch_refused(mesh_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from seiso.mesh.coordinator import build_plan, worker_env

    plan_out = build_plan(
        channel="ch-1",
        job_type="finetune",
        nodes=2,
        master_addr="10.0.0.2",
        gpus_per_node=1,
    )
    monkeypatch.setenv("SEISO_MESH_TOKEN", "other-token")
    with pytest.raises(RuntimeError, match="does not match"):
        worker_env(plan_out["plan"], node_rank=0)


def test_load_plan_refuses_foreign_absolute_path(
    mesh_env: Path, tmp_path: Path
) -> None:
    from seiso.mesh.coordinator import load_plan

    foreign = tmp_path / "evil-plan.json"
    foreign.write_text('{"job_id":"x","distributed_num_nodes":2}\n', encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        load_plan(str(foreign))


def test_build_plan_refuses_localhost_multinode(mesh_env: Path) -> None:
    from seiso.mesh.coordinator import build_plan

    with pytest.raises(ValueError, match="reachable multi-host"):
        build_plan(
            channel="ch-1",
            job_type="finetune",
            nodes=2,
            master_addr="127.0.0.1",
            gpus_per_node=1,
        )


def json_dumps(obj: object) -> str:
    import json

    return json.dumps(obj)
