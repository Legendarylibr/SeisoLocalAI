"""Experimental Buzz mesh — opt-in, no protocol fee."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture()
def mesh_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("SEISO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SEISO_ALLOW_MESH", "1")
    monkeypatch.setenv("SEISO_MESH_TOKEN", "test-mesh-token")
    return tmp_path / "data"


def test_mesh_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SEISO_ALLOW_MESH", raising=False)
    from seiso.mesh.flags import mesh_allowed, require_mesh_allowed

    assert mesh_allowed() is False
    with pytest.raises(RuntimeError, match="SEISO_ALLOW_MESH"):
        require_mesh_allowed()


def test_announce_plan_worker(mesh_env: Path) -> None:
    from seiso.mesh.coordinator import announce, build_plan, load_plan, worker_env

    ann = announce(channel="ch-1", gpus=2, capabilities=["finetune", "slime"])
    assert ann["buzz_receipt"]["role"] == "announce"
    assert "test-mesh-token" not in json_dumps(ann["buzz_receipt"])
    assert "token" not in ann["buzz_receipt"]

    plan_out = build_plan(
        channel="ch-1",
        job_type="finetune",
        nodes=2,
        master_addr="10.0.0.2",
    )
    assert plan_out["plan"]["protocol_fee_sats"] == 0
    assert plan_out["plan"]["market"] is False
    assert plan_out["buzz_receipt"]["world_size"] == 2

    plan = load_plan(plan_out["plan_path"])
    env = worker_env(plan, node_rank=1)
    assert env["NODE_RANK"] == "1"
    assert env["NNODES"] == "2"
    assert env["MASTER_ADDR"] == "10.0.0.2"


def json_dumps(obj: object) -> str:
    import json

    return json.dumps(obj)
