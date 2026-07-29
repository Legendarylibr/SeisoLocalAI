"""Mesh announce / plan / worker helpers (Buzz is the control plane).

Not functional yet — experimental scaffolding. Requires SEISO_ALLOW_MESH=1 and
a valid Buzz agent ``BUZZ_PRIVATE_KEY`` nsec. Plans/announces/heartbeats are
NIP-01 events signed with BIP-340 Schnorr. Peers also share an out-of-band
``SEISO_MESH_TOKEN`` (HMAC-bound per job+pubkey). Forge UI cannot start mesh.

Seiso does not NIP-98 to the Buzz relay (buzz-cli does). Post only
``buzz_receipt`` / ``agent_receipt`` to channels — never tokens or nsecs.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import socket
import time
import uuid
from pathlib import Path
from typing import Any, cast

from seiso.agent.nostr_identity import require_buzz_nsec
from seiso.agent.receipts import agent_receipt, channel_safe_plan_view
from seiso.mesh.flags import mesh_token, require_mesh_allowed
from seiso.mesh.nostr_bind import (
    receipt_nostr_fields,
    sign_mesh_announce,
    sign_mesh_heartbeat,
    sign_mesh_plan,
    verify_mesh_plan_nostr,
)
from seiso.security import SecurityError, resolve_data_dir, safe_join

# Weak shared secrets are dictionary-attackable if fingerprints ever leak.
_MIN_MESH_TOKEN_LEN = 16


def mesh_root(data_dir: Path | None = None) -> Path:
    root = resolve_data_dir(data_dir)
    path = safe_join(root, "mesh")
    path.mkdir(parents=True, exist_ok=True)
    (path / "plans").mkdir(parents=True, exist_ok=True)
    (path / "announces").mkdir(parents=True, exist_ok=True)
    return path


def _require_strong_mesh_token(token: str) -> str:
    raw = (token or "").strip()
    if len(raw) < _MIN_MESH_TOKEN_LEN:
        raise RuntimeError(
            f"SEISO_MESH_TOKEN must be at least {_MIN_MESH_TOKEN_LEN} characters "
            "(shared out-of-band secret; never post to Buzz)"
        )
    return raw


def _token_fingerprint(token: str, *, job_id: str, pubkey_hex: str) -> str:
    """Per-plan HMAC bound to job id + signing pubkey."""
    return hmac.new(
        token.encode("utf-8"),
        f"seiso-mesh-plan:{job_id}:{pubkey_hex.lower()}".encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def _verify_plan_bindings(plan: dict[str, Any]) -> None:
    """Verify Nostr signature + shared-secret fingerprint (fail closed)."""
    verify_mesh_plan_nostr(plan)
    expected = str(plan.get("token_fingerprint") or "").strip()
    token = _require_strong_mesh_token(mesh_token())
    if not expected:
        raise RuntimeError(
            "Plan is missing token_fingerprint — recreate with seiso mesh plan "
            "(unsigned / foreign plans are refused)"
        )
    job_id = str(plan.get("job_id") or "").strip()
    if not job_id:
        raise RuntimeError("Plan is missing job_id")
    pubkey = str((plan.get("nostr") or {}).get("pubkey") or "").strip()
    if not pubkey:
        raise RuntimeError("Plan is missing nostr.pubkey")
    actual = _token_fingerprint(token, job_id=job_id, pubkey_hex=pubkey)
    if not hmac.compare_digest(expected, actual):
        raise RuntimeError(
            "SEISO_MESH_TOKEN does not match this plan (shared-secret mismatch)"
        )


def announce(
    *,
    channel: str,
    gpus: int = 1,
    capabilities: list[str] | None = None,
    alias: str | None = None,
    data_dir: Path | None = None,
) -> dict[str, Any]:
    """Record a local announce + return Buzz-safe JSON (Nostr-signed, no secrets)."""
    require_mesh_allowed()
    pair = require_buzz_nsec(feature="Mesh announce")
    caps = capabilities or ["finetune", "slime"]
    record = {
        "role": "announce",
        "channel": channel,
        "gpus": int(gpus),
        "capabilities": caps,
        "alias": alias or socket.gethostname(),
        "hostname": socket.gethostname(),
        "ts": time.time(),
        "mesh_endpoint_fingerprint": uuid.uuid4().hex[:16],
    }
    nostr = sign_mesh_announce(record, pair)
    record["nostr"] = nostr
    path = safe_join(
        mesh_root(data_dir), "announces", f"{record['mesh_endpoint_fingerprint']}.json"
    )
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    receipt = agent_receipt(
        role="announce",
        status="announced",
        channel=channel,
        gpus=record["gpus"],
        capabilities=caps,
        alias=record["alias"],
        mesh_endpoint_fingerprint=record["mesh_endpoint_fingerprint"],
        **receipt_nostr_fields(nostr),
    )
    return {
        "buzz_receipt": receipt,
        "agent_receipt": receipt,
        "local_path": str(path),
        "nostr_event": nostr.get("event"),
        "note": (
            "Post buzz_receipt / agent_receipt to the channel; "
            "optionally post nostr_event via buzz-cli. "
            "Keep SEISO_MESH_TOKEN and nsec out-of-band."
        ),
    }


def build_plan(
    *,
    channel: str,
    job_type: str,
    nodes: int,
    preset: str | None = None,
    master_addr: str = "127.0.0.1",
    master_port: int = 29500,
    gpus_per_node: int | None = None,
    data_dir: Path | None = None,
) -> dict[str, Any]:
    """Create a multi-node plan mapped to Seiso Accelerate distributed_* knobs."""
    require_mesh_allowed()
    pair = require_buzz_nsec(feature="Mesh plan")
    if nodes < 1:
        raise ValueError("nodes must be >= 1")
    if nodes == 1:
        raise ValueError(
            "Mesh plans require nodes>=2 (single-node training uses Forge UI / "
            "`seiso train` without mesh)"
        )
    if gpus_per_node is None:
        raise ValueError(
            "gpus_per_node is required so every worker pins the same "
            "distributed_nproc_per_node (heterogeneous defaults disagree on world size)"
        )
    token = _require_strong_mesh_token(mesh_token())
    if master_addr.strip() in {"127.0.0.1", "localhost"} and nodes >= 2:
        raise ValueError(
            "distributed_master_addr must be a reachable multi-host address when "
            "nodes>=2 (refusing 127.0.0.1/localhost)"
        )
    job_id = uuid.uuid4().hex
    jt = job_type.strip().lower()
    if jt not in {"finetune", "slime"}:
        raise ValueError("mesh v1 supports job_type finetune|slime only")
    nproc = int(gpus_per_node)
    if nproc < 1:
        raise ValueError("gpus_per_node must be >= 1")
    plan: dict[str, Any] = {
        "job_id": job_id,
        "channel": channel,
        "job_type": jt,
        "preset": preset or "smoke",
        # Buzz receipt "world_size" = nodes; Accelerate world_size = nproc * nodes.
        "world_size_nodes": int(nodes),
        "distributed_num_nodes": int(nodes),
        "distributed_nproc_per_node": nproc,
        "distributed_master_addr": master_addr,
        "distributed_master_port": int(master_port),
        "distributed_strategy": "ddp",
        "multi_gpu": True,
        "protocol_fee_sats": 0,
        "market": False,
        "token_fingerprint": _token_fingerprint(
            token, job_id=job_id, pubkey_hex=pair.public_hex
        ),
        "created_at": time.time(),
        "ranks": [
            {
                "rank": i,
                "distributed_node_rank": i,
                "status": "pending",
            }
            for i in range(int(nodes))
        ],
    }
    sign_mesh_plan(plan, pair)
    path = safe_join(mesh_root(data_dir), "plans", f"{job_id}.json")
    path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    nostr = plan["nostr"]
    receipt = agent_receipt(
        role="plan",
        status="planned",
        job_id=job_id,
        type=jt,
        world_size=int(nodes),
        world_size_nodes=int(nodes),
        **receipt_nostr_fields(nostr),
    )
    return {
        "plan": plan,
        "plan_public": channel_safe_plan_view(plan),
        "plan_path": str(path),
        "buzz_receipt": receipt,
        "agent_receipt": receipt,
        "nostr_event": nostr.get("event"),
        "note": (
            "Post buzz_receipt / agent_receipt only (includes npub + event id). "
            "Optionally relay nostr_event via buzz-cli. Never post "
            "token_fingerprint, SEISO_MESH_TOKEN, or nsecs."
        ),
    }


def load_plan(plan_path: str | Path, data_dir: Path | None = None) -> dict[str, Any]:
    """Load a plan only from the sandboxed ``mesh/plans/`` directory.

    Accepts a job_id or a ``*.json`` basename. Absolute / foreign filesystem
    paths are refused (P1: no unsandboxed plan load).
    """
    plans_dir = mesh_root(data_dir) / "plans"
    raw = str(plan_path).strip()
    if not raw:
        raise FileNotFoundError(plan_path)
    name = Path(raw).name
    job_key = name[: -len(".json")] if name.endswith(".json") else name
    if not job_key or job_key in {".", ".."}:
        raise FileNotFoundError(plan_path)
    try:
        path = safe_join(plans_dir, f"{job_key}.json")
    except SecurityError as exc:
        raise FileNotFoundError(plan_path) from exc
    if not path.is_file():
        raise FileNotFoundError(plan_path)
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def worker_env(plan: dict[str, Any], *, node_rank: int) -> dict[str, str]:
    """Environment overlays for Accelerate multi-node worker (informational).

    Seiso train / Forge honor ``worker_train_config_overlay`` (TrainConfig), not
    these env vars alone. Env is kept for Accelerate-native launches.
    """
    require_mesh_allowed()
    _verify_plan_bindings(plan)
    nnodes = int(plan["distributed_num_nodes"])
    if node_rank < 0 or node_rank >= nnodes:
        raise ValueError("node_rank out of range")
    env = {
        "SEISO_MESH_JOB_ID": str(plan["job_id"]),
        # Flag presence only — not a secret (bandit B105 false positive).
        "SEISO_MESH_TOKEN_SET": "1",  # nosec B105
        "SEISO_TRAINING_SURFACE": "agent",
        "SEISO_MESH_PLANNER_NPUB": str((plan.get("nostr") or {}).get("npub") or ""),
        "MASTER_ADDR": str(plan["distributed_master_addr"]),
        "MASTER_PORT": str(plan["distributed_master_port"]),
        "NNODES": str(nnodes),
        "NODE_RANK": str(node_rank),
    }
    nproc = plan.get("distributed_nproc_per_node")
    if nproc is not None:
        env["NPROC_PER_NODE"] = str(int(nproc))
    return env


def worker_train_config_overlay(plan: dict[str, Any], *, node_rank: int) -> dict[str, Any]:
    """TrainConfig fields applied by agents — includes nproc when planned."""
    require_mesh_allowed()
    _verify_plan_bindings(plan)
    nnodes = int(plan["distributed_num_nodes"])
    if node_rank < 0 or node_rank >= nnodes:
        raise ValueError("node_rank out of range")
    overlay: dict[str, Any] = {
        "multi_gpu": True,
        "distributed_strategy": "ddp",
        "distributed_num_nodes": nnodes,
        "distributed_node_rank": int(node_rank),
        "distributed_master_addr": plan["distributed_master_addr"],
        "distributed_master_port": int(plan["distributed_master_port"]),
    }
    nproc = plan.get("distributed_nproc_per_node")
    if nproc is not None:
        overlay["distributed_nproc_per_node"] = int(nproc)
    return overlay


def buzz_heartbeat(plan: dict[str, Any], *, node_rank: int, status: str) -> dict[str, Any]:
    pair = require_buzz_nsec(feature="Mesh heartbeat")
    nostr = sign_mesh_heartbeat(
        plan=plan, node_rank=node_rank, status=status, pair=pair
    )
    receipt = agent_receipt(
        role="heartbeat",
        status=status,
        job_id=plan.get("job_id"),
        type=plan.get("job_type"),
        rank=node_rank,
        world_size=plan.get("distributed_num_nodes"),
        world_size_nodes=plan.get("distributed_num_nodes"),
        **receipt_nostr_fields(nostr),
    )
    return receipt
