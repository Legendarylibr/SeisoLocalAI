"""Mesh announce / plan / worker helpers (Buzz is the control plane)."""

from __future__ import annotations

import json
import os
import socket
import time
import uuid
from pathlib import Path
from typing import Any

from seiso.mesh.flags import mesh_token, require_mesh_allowed
from seiso.security import resolve_data_dir, safe_join


def mesh_root(data_dir: Path | None = None) -> Path:
    root = resolve_data_dir(data_dir)
    path = safe_join(root, "mesh")
    path.mkdir(parents=True, exist_ok=True)
    (path / "plans").mkdir(parents=True, exist_ok=True)
    (path / "announces").mkdir(parents=True, exist_ok=True)
    return path


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
    return {
        "buzz_receipt": {
            "role": "announce",
            "channel": channel,
            "gpus": record["gpus"],
            "capabilities": caps,
            "alias": record["alias"],
            "mesh_endpoint_fingerprint": record["mesh_endpoint_fingerprint"],
        },
        "local_path": str(path),
        "note": "Post buzz_receipt to the channel; keep SEISO_MESH_TOKEN out-of-band",
    }


def build_plan(
    *,
    channel: str,
    job_type: str,
    nodes: int,
    preset: str | None = None,
    master_addr: str = "127.0.0.1",
    master_port: int = 29500,
    data_dir: Path | None = None,
) -> dict[str, Any]:
    """Create a multi-node plan mapped to Seiso Accelerate distributed_* knobs."""
    require_mesh_allowed()
    if nodes < 1:
        raise ValueError("nodes must be >= 1")
    if not mesh_token():
        raise RuntimeError(
            "SEISO_MESH_TOKEN required out-of-band for workers (never post to Buzz)"
        )
    job_id = uuid.uuid4().hex
    jt = job_type.strip().lower()
    if jt not in {"finetune", "slime"}:
        raise ValueError("mesh v1 supports job_type finetune|slime only")
    plan = {
        "job_id": job_id,
        "channel": channel,
        "job_type": jt,
        "preset": preset or "smoke",
        "world_size_nodes": int(nodes),
        "distributed_num_nodes": int(nodes),
        "distributed_master_addr": master_addr,
        "distributed_master_port": int(master_port),
        "distributed_strategy": "ddp",
        "multi_gpu": True,
        "protocol_fee_sats": 0,
        "market": False,
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
    buzz = {
        "role": "plan",
        "job_id": job_id,
        "type": jt,
        "world_size": int(nodes),
        "status": "planned",
        "master_hint": master_addr,
        # no token, no private dataset paths
    }
    return {"plan": plan, "plan_path": str(path), "buzz_receipt": buzz}


def load_plan(plan_path: str | Path, data_dir: Path | None = None) -> dict[str, Any]:
    path = Path(plan_path)
    if not path.is_file():
        # try job_id
        candidate = safe_join(mesh_root(data_dir), "plans", f"{plan_path}.json")
        if candidate.is_file():
            path = candidate
        else:
            raise FileNotFoundError(plan_path)
    return json.loads(path.read_text(encoding="utf-8"))


def worker_env(plan: dict[str, Any], *, node_rank: int) -> dict[str, str]:
    """Environment / config overlays for Accelerate multi-node worker."""
    require_mesh_allowed()
    if not mesh_token():
        raise RuntimeError("SEISO_MESH_TOKEN required")
    if node_rank < 0 or node_rank >= int(plan["distributed_num_nodes"]):
        raise ValueError("node_rank out of range")
    return {
        "SEISO_MESH_JOB_ID": str(plan["job_id"]),
        "SEISO_MESH_TOKEN_SET": "1",
        "MASTER_ADDR": str(plan["distributed_master_addr"]),
        "MASTER_PORT": str(plan["distributed_master_port"]),
        "NNODES": str(plan["distributed_num_nodes"]),
        "NODE_RANK": str(node_rank),
    }


def worker_train_config_overlay(plan: dict[str, Any], *, node_rank: int) -> dict[str, Any]:
    return {
        "multi_gpu": True,
        "distributed_strategy": "ddp",
        "distributed_num_nodes": int(plan["distributed_num_nodes"]),
        "distributed_node_rank": int(node_rank),
        "distributed_master_addr": plan["distributed_master_addr"],
        "distributed_master_port": int(plan["distributed_master_port"]),
    }


def buzz_heartbeat(
    plan: dict[str, Any], *, node_rank: int, status: str
) -> dict[str, Any]:
    return {
        "role": "heartbeat",
        "job_id": plan.get("job_id"),
        "type": plan.get("job_type"),
        "rank": node_rank,
        "world_size": plan.get("distributed_num_nodes"),
        "status": status,
    }
