"""Mesh announce / plan / worker helpers (Buzz is the control plane).

Not functional yet — experimental scaffolding. Requires SEISO_ALLOW_MESH=1 and
a Buzz agent identity (BUZZ_PRIVATE_KEY / BUZZ_AUTH_TAG). Forge UI cannot start
mesh jobs; use a Buzz agent chat / CLI instead.
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

from seiso.agent.receipts import agent_receipt
from seiso.mesh.flags import mesh_token, require_mesh_allowed
from seiso.security import SecurityError, resolve_data_dir, safe_join


def mesh_root(data_dir: Path | None = None) -> Path:
    root = resolve_data_dir(data_dir)
    path = safe_join(root, "mesh")
    path.mkdir(parents=True, exist_ok=True)
    (path / "plans").mkdir(parents=True, exist_ok=True)
    (path / "announces").mkdir(parents=True, exist_ok=True)
    return path


def _token_fingerprint(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _verify_plan_token(plan: dict[str, Any]) -> None:
    """Fail closed unless the live SEISO_MESH_TOKEN matches the plan fingerprint."""
    expected = str(plan.get("token_fingerprint") or "").strip()
    token = mesh_token()
    if not token:
        raise RuntimeError("SEISO_MESH_TOKEN required")
    if not expected:
        raise RuntimeError(
            "Plan is missing token_fingerprint — recreate with seiso mesh plan "
            "(unsigned / foreign plans are refused)"
        )
    actual = _token_fingerprint(token)
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
    """Record a local announce + return Buzz-safe JSON (no secrets)."""
    require_mesh_allowed()
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
        # Never include SEISO_MESH_TOKEN or private IPs here for channel posts.
    }
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
    )
    return {
        "buzz_receipt": receipt,
        "agent_receipt": receipt,
        "local_path": str(path),
        "note": (
            "Post buzz_receipt / agent_receipt to the channel; "
            "keep SEISO_MESH_TOKEN out-of-band"
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
    token = mesh_token()
    if not token:
        raise RuntimeError(
            "SEISO_MESH_TOKEN required out-of-band for workers (never post to Buzz)"
        )
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
        "token_fingerprint": _token_fingerprint(token),
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
    path = safe_join(mesh_root(data_dir), "plans", f"{job_id}.json")
    path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    receipt = agent_receipt(
        role="plan",
        status="planned",
        job_id=job_id,
        type=jt,
        # Node count for Buzz room semantics (not process world size).
        world_size=int(nodes),
        world_size_nodes=int(nodes),
        master_hint=master_addr,
    )
    return {
        "plan": plan,
        "plan_path": str(path),
        "buzz_receipt": receipt,
        "agent_receipt": receipt,
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
    _verify_plan_token(plan)
    nnodes = int(plan["distributed_num_nodes"])
    if node_rank < 0 or node_rank >= nnodes:
        raise ValueError("node_rank out of range")
    env = {
        "SEISO_MESH_JOB_ID": str(plan["job_id"]),
        # Flag presence only — not a secret (bandit B105 false positive).
        "SEISO_MESH_TOKEN_SET": "1",  # nosec B105
        "SEISO_TRAINING_SURFACE": "agent",
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
    _verify_plan_token(plan)
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
    receipt = agent_receipt(
        role="heartbeat",
        status=status,
        job_id=plan.get("job_id"),
        type=plan.get("job_type"),
        rank=node_rank,
        world_size=plan.get("distributed_num_nodes"),
        world_size_nodes=plan.get("distributed_num_nodes"),
    )
    return receipt
