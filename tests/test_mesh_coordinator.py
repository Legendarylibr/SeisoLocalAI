"""Experimental Buzz mesh — Nostr-signed, Buzz-agent-only, no protocol fee."""

from __future__ import annotations

from pathlib import Path

import pytest

from seiso.research.nostr.events import verify_event
from seiso.research.nostr.keys import generate_keypair


@pytest.fixture()
def mesh_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("SEISO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SEISO_ALLOW_MESH", "1")
    monkeypatch.setenv("SEISO_MESH_TOKEN", "test-mesh-token-ok")
    monkeypatch.setenv("BUZZ_PRIVATE_KEY", generate_keypair().nsec)
    monkeypatch.setenv("SEISO_AGENT", "1")
    monkeypatch.delenv("SEISO_MESH_TRUSTED_NPUBS", raising=False)
    monkeypatch.delenv("SEISO_MESH_TRUSTED_PUBKEYS", raising=False)
    return tmp_path / "data"


def test_mesh_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SEISO_ALLOW_MESH", raising=False)
    monkeypatch.setenv("BUZZ_PRIVATE_KEY", generate_keypair().nsec)
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
    assert ann["buzz_receipt"]["npub"].startswith("npub1")
    assert ann["buzz_receipt"]["nostr_event_id"]
    assert ann["buzz_receipt"]["sig_alg"] == "bip340-schnorr"
    assert ann["buzz_receipt"]["relay_policy"] == "signed_event_only"
    assert "Relay policy" in ann["note"]
    assert verify_event(ann["nostr_event"])
    assert ann["nostr_event"]["sig"]
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
    assert "token_fingerprint" not in plan_out["plan"]["nostr"]["event"]["content"]
    assert plan_out["plan"]["nostr"]["event_id"]
    assert verify_event(plan_out["plan"]["nostr"]["event"])
    assert verify_event(plan_out["nostr_event"])
    assert "token_fingerprint" not in plan_out["plan_public"]
    assert "token_fingerprint" not in (plan_out["nostr_event"].get("content") or "")
    assert "event" not in (plan_out["plan_public"].get("nostr") or {})
    assert plan_out["buzz_receipt"]["world_size"] == 2
    assert plan_out["buzz_receipt"]["npub"].startswith("npub1")
    assert plan_out["buzz_receipt"]["nostr_kind"] == 31251

    plan = load_plan(plan_out["plan"]["job_id"])
    env = worker_env(plan, node_rank=1)
    assert env["NODE_RANK"] == "1"
    assert env["NNODES"] == "2"
    assert env["MASTER_ADDR"] == "10.0.0.2"
    assert env["NPROC_PER_NODE"] == "2"
    assert env["SEISO_MESH_PLANNER_NPUB"].startswith("npub1")
    overlay = worker_train_config_overlay(plan, node_rank=1)
    assert overlay["distributed_nproc_per_node"] == 2
    assert overlay["distributed_node_rank"] == 1

    from seiso.mesh.coordinator import buzz_heartbeat

    hb = buzz_heartbeat(plan, node_rank=1, status="joining")
    assert hb["buzz_receipt"]["nostr_event_id"] == hb["agent_receipt"]["nostr_event_id"]
    assert hb["nostr_event"]["id"] == hb["buzz_receipt"]["nostr_event_id"]
    assert "nostr_event" not in hb["buzz_receipt"]


def test_event_id_wrapper_mismatch_refused(mesh_env: Path) -> None:
    from seiso.mesh.coordinator import build_plan, worker_env

    plan_out = build_plan(
        channel="ch-1",
        job_type="finetune",
        nodes=2,
        master_addr="10.0.0.2",
        gpus_per_node=1,
    )
    plan = plan_out["plan"]
    plan["nostr"]["event_id"] = "ab" * 32
    with pytest.raises(RuntimeError, match="event_id"):
        worker_env(plan, node_rank=0)


def test_tampered_plan_nostr_refused(mesh_env: Path) -> None:
    from seiso.mesh.coordinator import build_plan, worker_env

    plan_out = build_plan(
        channel="ch-1",
        job_type="finetune",
        nodes=2,
        master_addr="10.0.0.2",
        gpus_per_node=1,
    )
    plan = plan_out["plan"]
    plan["distributed_num_nodes"] = 99
    with pytest.raises(RuntimeError, match="tamper|Nostr|signature|match"):
        worker_env(plan, node_rank=0)


def test_trusted_npub_allowlist(
    mesh_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from seiso.mesh.coordinator import build_plan, worker_env

    plan_out = build_plan(
        channel="ch-1",
        job_type="finetune",
        nodes=2,
        master_addr="10.0.0.2",
        gpus_per_node=1,
    )
    planner_npub = plan_out["plan"]["nostr"]["npub"]
    monkeypatch.setenv("SEISO_MESH_TRUSTED_NPUBS", planner_npub)
    worker_env(plan_out["plan"], node_rank=0)

    other = generate_keypair()
    monkeypatch.setenv("SEISO_MESH_TRUSTED_NPUBS", other.npub)
    with pytest.raises(RuntimeError, match="TRUSTED"):
        worker_env(plan_out["plan"], node_rank=0)


def test_relay_signed_event_refuses_unsigned() -> None:
    from seiso.mesh.nostr_bind import relay_signed_event

    with pytest.raises(RuntimeError, match="Relay only with signing"):
        relay_signed_event({})
    with pytest.raises(RuntimeError, match="Relay only with signing"):
        relay_signed_event({"event": {"id": "ab" * 32, "sig": "00" * 64}})


def test_build_plan_requires_gpus_per_node(mesh_env: Path) -> None:
    from seiso.mesh.coordinator import build_plan

    with pytest.raises(ValueError, match="gpus_per_node is required"):
        build_plan(
            channel="ch-1",
            job_type="finetune",
            nodes=2,
            master_addr="10.0.0.2",
        )


def test_mesh_token_mismatch_refused(
    mesh_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from seiso.mesh.coordinator import build_plan, worker_env

    plan_out = build_plan(
        channel="ch-1",
        job_type="finetune",
        nodes=2,
        master_addr="10.0.0.2",
        gpus_per_node=1,
    )
    monkeypatch.setenv("SEISO_MESH_TOKEN", "other-token-16chars")
    with pytest.raises(RuntimeError, match="does not match"):
        worker_env(plan_out["plan"], node_rank=0)


def test_short_mesh_token_refused(
    mesh_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from seiso.mesh.coordinator import build_plan

    monkeypatch.setenv("SEISO_MESH_TOKEN", "short")
    with pytest.raises(RuntimeError, match="at least 16"):
        build_plan(
            channel="ch-1",
            job_type="finetune",
            nodes=2,
            master_addr="10.0.0.2",
            gpus_per_node=1,
        )


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
