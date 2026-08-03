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
    # Unit tests are single-operator smoke — production requires trusted npubs.
    monkeypatch.setenv("SEISO_MESH_ALLOW_ANY_PLANNER", "1")
    monkeypatch.setenv("BUZZ_PRIVATE_KEY", generate_keypair().nsec)
    monkeypatch.setenv("SEISO_AGENT", "1")
    monkeypatch.delenv("SEISO_MESH_TRUSTED_NPUBS", raising=False)
    monkeypatch.delenv("SEISO_MESH_TRUSTED_PUBKEYS", raising=False)
    monkeypatch.delenv("SEISO_MESH_ALLOW_LOOPBACK", raising=False)
    monkeypatch.delenv("SEISO_MESH_CONFIRM_LAUNCH", raising=False)
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
    # Privacy: default alias is opaque peer-*, never OS hostname.
    import json as _json
    import socket as _socket

    ann_body = _json.loads(ann["nostr_event"]["content"])
    assert ann_body["alias"].startswith("peer-")
    assert ann_body["alias"] != _socket.gethostname()
    assert "hostname" not in ann_body
    assert "hostname" not in ann["buzz_receipt"]
    assert ann["buzz_receipt"]["alias"] == ann_body["alias"]

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


def test_missing_event_id_refused(mesh_env: Path) -> None:
    from seiso.mesh.coordinator import build_plan, worker_env

    plan_out = build_plan(
        channel="ch-1",
        job_type="finetune",
        nodes=2,
        master_addr="10.0.0.2",
        gpus_per_node=1,
    )
    plan = plan_out["plan"]
    plan["nostr"]["event_id"] = ""
    with pytest.raises(RuntimeError, match="missing nostr.event_id"):
        worker_env(plan, node_rank=0)


def test_heartbeat_requires_verified_plan(mesh_env: Path) -> None:
    from seiso.mesh.coordinator import build_plan, buzz_heartbeat

    plan_out = build_plan(
        channel="ch-1",
        job_type="finetune",
        nodes=2,
        master_addr="10.0.0.2",
        gpus_per_node=1,
    )
    plan = dict(plan_out["plan"])
    plan["distributed_num_nodes"] = 99
    with pytest.raises(RuntimeError, match="tamper|Nostr|signature|match"):
        buzz_heartbeat(plan, node_rank=0, status="joining")


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
    monkeypatch.delenv("SEISO_MESH_ALLOW_ANY_PLANNER", raising=False)
    monkeypatch.setenv("SEISO_MESH_TRUSTED_NPUBS", planner_npub)
    worker_env(plan_out["plan"], node_rank=0)

    other = generate_keypair()
    monkeypatch.setenv("SEISO_MESH_TRUSTED_NPUBS", other.npub)
    with pytest.raises(RuntimeError, match="TRUSTED"):
        worker_env(plan_out["plan"], node_rank=0)


def test_empty_trusted_allowlist_refused_without_opt_out(
    mesh_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SEISO_MESH_ALLOW_ANY_PLANNER", raising=False)
    monkeypatch.delenv("SEISO_MESH_TRUSTED_NPUBS", raising=False)
    monkeypatch.delenv("SEISO_MESH_TRUSTED_PUBKEYS", raising=False)
    from seiso.mesh.coordinator import build_plan

    with pytest.raises(RuntimeError, match="TRUSTED_NPUBS|ALLOW_ANY_PLANNER"):
        build_plan(
            channel="ch-1",
            job_type="finetune",
            nodes=2,
            master_addr="10.0.0.2",
            gpus_per_node=1,
        )


def test_launch_requires_confirm(
    mesh_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SEISO_MESH_ALLOW_LOOPBACK", "1")
    from seiso.mesh.coordinator import build_plan, prepare_worker

    plan_out = build_plan(
        channel="ch-1",
        job_type="finetune",
        nodes=2,
        master_addr="127.0.0.1",
        gpus_per_node=1,
    )
    base = tmp_path / "base.yaml"
    base.write_text(
        "model_id: hf-internal-testing/tiny-random-LlamaForCausalLM\n"
        "dataset: ./data/sample.jsonl\noutput_dir: ./out\nmethod: lora\nquant: 16bit\n"
        "epochs: 1\nbatch_size: 1\nmax_seq_length: 128\nlora_r: 4\nlora_alpha: 8\n"
        "gradient_checkpointing: false\neval_split_ratio: 0\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="confirm"):
        prepare_worker(
            plan_out["plan"]["job_id"],
            node_rank=0,
            base_config=base,
            launch=True,
            confirm_launch=False,
        )


def test_import_signed_plan_refuses_bad_job_type(mesh_env: Path) -> None:
    import json
    import os

    from seiso.mesh.coordinator import build_plan, import_signed_plan
    from seiso.research.nostr.events import sign_event
    from seiso.research.nostr.keys import keypair_from_secret

    plan_out = build_plan(
        channel="ch-1",
        job_type="finetune",
        nodes=2,
        master_addr="10.0.0.2",
        gpus_per_node=1,
    )
    event = dict(plan_out["nostr_event"])
    body = json.loads(event["content"])
    body["job_type"] = "evil"
    pair = keypair_from_secret(os.environ["BUZZ_PRIVATE_KEY"])
    draft = {
        "kind": event["kind"],
        "created_at": event["created_at"],
        "tags": event["tags"],
        "content": json.dumps(body, separators=(",", ":"), sort_keys=True),
    }
    evil = sign_event(draft, pair)
    with pytest.raises(ValueError, match="job_type"):
        import_signed_plan(evil)


def test_relay_signed_event_refuses_unsigned() -> None:
    from seiso.mesh.nostr_bind import relay_signed_event

    with pytest.raises(RuntimeError, match="Relay only with signing"):
        relay_signed_event({})
    with pytest.raises(RuntimeError, match="Relay only with signing"):
        relay_signed_event({"event": {"id": "ab" * 32, "sig": "00" * 64}})


def test_sign_mesh_announce_requires_d_tag() -> None:
    from seiso.mesh.nostr_bind import sign_mesh_announce
    from seiso.research.nostr.keys import generate_keypair

    pair = generate_keypair()
    with pytest.raises(RuntimeError, match="mesh_endpoint_fingerprint"):
        sign_mesh_announce(
            {
                "channel": "ch",
                "gpus": 1,
                "capabilities": [],
                "alias": "x",
                "mesh_endpoint_fingerprint": "",
                "ts": 1,
            },
            pair,
        )


def test_sign_mesh_announce_refuses_hostname_alias() -> None:
    import socket

    from seiso.mesh.nostr_bind import sign_mesh_announce
    from seiso.research.nostr.keys import generate_keypair

    pair = generate_keypair()
    host = socket.gethostname()
    with pytest.raises(RuntimeError, match="hostname"):
        sign_mesh_announce(
            {
                "channel": "ch",
                "gpus": 1,
                "capabilities": [],
                "alias": host,
                "hostname": host,
                "mesh_endpoint_fingerprint": "abc123def4567890",
                "ts": 1,
            },
            pair,
        )


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

    with pytest.raises(ValueError, match="reachable multi-host|SEISO_MESH_ALLOW_LOOPBACK"):
        build_plan(
            channel="ch-1",
            job_type="finetune",
            nodes=2,
            master_addr="127.0.0.1",
            gpus_per_node=1,
        )


def test_build_plan_allows_loopback_when_opted_in(
    mesh_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SEISO_MESH_ALLOW_LOOPBACK", "1")
    from seiso.mesh.coordinator import build_plan

    out = build_plan(
        channel="ch-1",
        job_type="finetune",
        nodes=2,
        master_addr="127.0.0.1",
        gpus_per_node=1,
    )
    assert out["plan"]["distributed_master_addr"] == "127.0.0.1"


def test_import_claim_materialize_dry_run_e2e(
    mesh_env: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Full secondary-path loop without requiring GPUs or real peers."""
    monkeypatch.setenv("SEISO_MESH_ALLOW_LOOPBACK", "1")
    from seiso.mesh.coordinator import (
        build_plan,
        import_signed_plan,
        load_plan,
        prepare_worker,
    )
    from seiso.training.config import TrainConfig

    plan_out = build_plan(
        channel="ch-e2e",
        job_type="finetune",
        nodes=2,
        master_addr="127.0.0.1",
        gpus_per_node=1,
        preset="smoke",
    )
    event = plan_out["nostr_event"]
    job_id = plan_out["plan"]["job_id"]

    # Simulate peer: wipe local plan file, re-import from signed event only.
    plan_path = Path(plan_out["plan_path"])
    plan_path.unlink()
    imported = import_signed_plan(event)
    assert imported["plan"]["job_id"] == job_id
    assert Path(imported["plan_path"]).is_file()
    assert verify_event(imported["nostr_event"])

    base = tmp_path / "base_train.yaml"
    base.write_text(
        "\n".join(
            [
                "model_id: hf-internal-testing/tiny-random-LlamaForCausalLM",
                "dataset: ./data/sample.jsonl",
                "output_dir: ./.test_outputs/mesh-e2e",
                "method: lora",
                "quant: 16bit",
                "epochs: 1",
                "batch_size: 1",
                "max_seq_length: 128",
                "lora_r: 4",
                "lora_alpha: 8",
                "gradient_checkpointing: false",
                "eval_split_ratio: 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    prepared = prepare_worker(
        job_id,
        node_rank=1,
        base_config=base,
        dry_run=True,
    )
    assert prepared["rank"] == 1
    assert prepared["config_path"]
    assert prepared["launch"]["dry_run"] is True
    assert prepared["launch"]["command"][-1] == prepared["config_path"]
    assert "train" in prepared["launch"]["command"]

    cfg = TrainConfig.from_yaml(Path(prepared["config_path"]))
    assert cfg.distributed_num_nodes == 2
    assert cfg.distributed_node_rank == 1
    assert cfg.distributed_master_addr == "127.0.0.1"
    assert cfg.distributed_nproc_per_node == 1
    assert cfg.multi_gpu is True

    reloaded = load_plan(job_id)
    assert reloaded["ranks"][1]["status"] == "claimed"
    assert reloaded["ranks"][1]["claimed_by"].startswith("npub1")
    assert reloaded["ranks"][0]["status"] == "pending"


def test_materialize_refuses_job_type_method_mismatch(
    mesh_env: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("SEISO_MESH_ALLOW_LOOPBACK", "1")
    from seiso.mesh.coordinator import build_plan, materialize_worker_config

    plan_out = build_plan(
        channel="ch-mismatch",
        job_type="finetune",
        nodes=2,
        master_addr="127.0.0.1",
        gpus_per_node=1,
    )
    base = tmp_path / "slime_base.yaml"
    base.write_text(
        "\n".join(
            [
                "model_id: hf-internal-testing/tiny-random-LlamaForCausalLM",
                "dataset: ./data/sample.jsonl",
                "output_dir: ./.test_outputs/mesh-mismatch",
                "method: slime",
                "quant: 16bit",
                "epochs: 1",
                "batch_size: 1",
                "max_seq_length: 128",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="job_type=finetune"):
        materialize_worker_config(
            plan_out["plan"],
            node_rank=0,
            base_config=base,
        )


def test_buzz_kind9_content_embed_roundtrip(mesh_env: Path) -> None:
    """Buzz rejects --kind 31251; peers embed the signed event as kind-9 content."""
    import json

    from seiso.mesh.coordinator import build_plan, import_signed_plan
    from seiso.mesh.nostr_bind import SEISO_MESH_PLAN_KIND

    plan_out = build_plan(
        channel="ch-buzz",
        job_type="finetune",
        nodes=2,
        master_addr="10.0.0.9",
        gpus_per_node=1,
    )
    event = plan_out["nostr_event"]
    assert event["kind"] == SEISO_MESH_PLAN_KIND
    assert verify_event(event)
    # Simulate `jq -c .nostr_event | buzz messages send --content -`
    kind9_content = json.dumps(event, separators=(",", ":"), ensure_ascii=False)
    assert "token_fingerprint" not in kind9_content
    assert "SEISO_MESH_TOKEN" not in kind9_content
    peer_event = json.loads(kind9_content)
    assert verify_event(peer_event)
    Path(plan_out["plan_path"]).unlink()
    imported = import_signed_plan(peer_event)
    assert imported["plan"]["job_id"] == plan_out["plan"]["job_id"]
    assert imported["nostr_event"]["id"] == event["id"]


def test_claim_rank_refuses_foreign_claimer(
    mesh_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SEISO_MESH_ALLOW_LOOPBACK", "1")
    from seiso.mesh.coordinator import build_plan, claim_rank, load_plan

    plan_out = build_plan(
        channel="ch-1",
        job_type="finetune",
        nodes=2,
        master_addr="127.0.0.1",
        gpus_per_node=1,
    )
    job_id = plan_out["plan"]["job_id"]
    claim_rank(plan_out["plan"], node_rank=0)

    other = generate_keypair()
    monkeypatch.setenv("BUZZ_PRIVATE_KEY", other.nsec)
    with pytest.raises(RuntimeError, match="already claimed"):
        claim_rank(load_plan(job_id), node_rank=0)


def test_import_signed_plan_refuses_bad_kind(mesh_env: Path) -> None:
    from seiso.mesh.coordinator import build_plan, import_signed_plan

    plan_out = build_plan(
        channel="ch-1",
        job_type="finetune",
        nodes=2,
        master_addr="10.0.0.2",
        gpus_per_node=1,
    )
    event = dict(plan_out["nostr_event"])
    event["kind"] = 1
    with pytest.raises(RuntimeError, match="expected kind|signature"):
        import_signed_plan(event)


def test_prepare_worker_launch_requires_base_config(mesh_env: Path) -> None:
    from seiso.mesh.coordinator import build_plan, prepare_worker

    plan_out = build_plan(
        channel="ch-1",
        job_type="finetune",
        nodes=2,
        master_addr="10.0.0.2",
        gpus_per_node=1,
    )
    with pytest.raises(ValueError, match="--base-config"):
        prepare_worker(plan_out["plan"]["job_id"], node_rank=0, launch=True)


def json_dumps(obj: object) -> str:
    import json

    return json.dumps(obj)
