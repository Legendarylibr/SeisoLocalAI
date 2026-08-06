"""Mesh announce / plan / worker helpers (Buzz is the control plane).

Experimental **secondary** multi-node path. Requires SEISO_ALLOW_MESH=1 and
a valid Buzz agent ``BUZZ_PRIVATE_KEY`` nsec. Plans/announces/heartbeats are
NIP-01 events signed with BIP-340 Schnorr. Peers also share an out-of-band
``SEISO_MESH_TOKEN`` (HMAC-bound per job+pubkey). Forge UI cannot start mesh;
local single-node training stays the primary path.

**Relay only with signing:** channel/relay authority is the signed
``nostr_event``. Unsigned receipts are local pointers. Seiso does not NIP-98
to the Buzz relay — buzz-cli publishes the signed event.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import socket
import time
import uuid
from pathlib import Path
from typing import Any, cast

import yaml

from seiso.agent.nostr_identity import require_buzz_nsec
from seiso.agent.receipts import agent_receipt, channel_safe_plan_view
from seiso.mesh.flags import (
    mesh_allow_loopback,
    mesh_confirm_launch,
    mesh_debug_local,
    mesh_token,
    require_mesh_allowed,
    require_mesh_planner_allowlist,
)
from seiso.mesh.nostr_bind import (
    SEISO_MESH_PLAN_KIND,
    receipt_nostr_fields,
    relay_policy_note,
    relay_signed_event,
    sign_mesh_announce,
    sign_mesh_heartbeat,
    sign_mesh_plan,
    verify_mesh_plan_nostr,
)
from seiso.research.nostr.events import verify_event
from seiso.research.nostr.keys import npub_from_hex
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
        token.encode(),
        f"seiso-mesh-plan:{job_id}:{pubkey_hex.lower()}".encode(),
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
        raise RuntimeError("SEISO_MESH_TOKEN does not match this plan (shared-secret mismatch)")


def announce(
    *,
    channel: str,
    gpus: int = 1,
    capabilities: list[str] | None = None,
    alias: str | None = None,
    data_dir: Path | None = None,
) -> dict[str, Any]:
    """Record a local announce + return Buzz-safe JSON (Nostr-signed, no secrets).

    Privacy: machine hostname stays on the local disk record only. Signed /
    channel-facing payloads use an explicit ``alias`` or an opaque
    ``peer-<fingerprint>`` — never the OS hostname (which would leak into Buzz
    when the signed event is embedded in kind-9).
    """
    require_mesh_allowed()
    pair = require_buzz_nsec(feature="Mesh announce")
    caps = capabilities or ["finetune", "slime"]
    fingerprint = uuid.uuid4().hex[:16]
    # Opaque peer label by default — do not publish OS hostname to the channel.
    public_alias = (alias or "").strip() or f"peer-{fingerprint}"
    record: dict[str, Any] = {
        "role": "announce",
        "channel": channel,
        "gpus": int(gpus),
        "capabilities": caps,
        "alias": public_alias,
        "ts": time.time(),
        "mesh_endpoint_fingerprint": fingerprint,
    }
    # Hostname stays off disk by default (debug-only) so a mistaken file paste
    # cannot leak machine identity into Buzz.
    if mesh_debug_local():
        record["hostname"] = socket.gethostname()
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
        "nostr_event": relay_signed_event(nostr),
        "note": relay_policy_note(),
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
    require_mesh_planner_allowlist()
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
    if (
        master_addr.strip() in {"127.0.0.1", "localhost"}
        and nodes >= 2
        and not mesh_allow_loopback()
    ):
        raise ValueError(
            "distributed_master_addr must be a reachable multi-host address when "
            "nodes>=2 (refusing 127.0.0.1/localhost). For single-host smoke only, "
            "set SEISO_MESH_ALLOW_LOOPBACK=1."
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
        "token_fingerprint": _token_fingerprint(token, job_id=job_id, pubkey_hex=pair.public_hex),
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
        "nostr_event": relay_signed_event(nostr),
        "note": (
            f"{relay_policy_note()} Keep token_fingerprint, SEISO_MESH_TOKEN, and nsecs local-only."
        ),
    }


def _plan_path(job_id: str, data_dir: Path | None = None) -> Path:
    return safe_join(mesh_root(data_dir), "plans", f"{job_id}.json")


def save_plan(plan: dict[str, Any], data_dir: Path | None = None) -> Path:
    """Persist a verified plan under the sandboxed ``mesh/plans/`` directory."""
    job_id = str(plan.get("job_id") or "").strip()
    if not job_id:
        raise ValueError("Plan is missing job_id")
    path = _plan_path(job_id, data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    return path


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


def import_signed_plan(
    event: dict[str, Any],
    *,
    data_dir: Path | None = None,
) -> dict[str, Any]:
    """Import a relayed NIP-01 mesh plan event into the local sandbox.

    Reconstructs local-only ``token_fingerprint`` and pending ranks from the
    signed body + ``SEISO_MESH_TOKEN``. Peers never receive the HMAC over Buzz.
    """
    require_mesh_allowed()
    require_mesh_planner_allowlist()
    if not isinstance(event, dict) or not verify_event(event):
        raise RuntimeError("Refusing import: mesh plan Nostr event signature is invalid")
    if int(event.get("kind") or 0) != SEISO_MESH_PLAN_KIND:
        raise RuntimeError(
            f"Refusing import: expected kind {SEISO_MESH_PLAN_KIND}, got {event.get('kind')}"
        )
    try:
        body = json.loads(str(event.get("content") or ""))
    except json.JSONDecodeError as exc:
        raise RuntimeError("Mesh plan Nostr event content is not JSON") from exc
    if not isinstance(body, dict):
        raise RuntimeError("Mesh plan Nostr event content must be a JSON object")
    job_id = str(body.get("job_id") or "").strip()
    if not job_id:
        raise RuntimeError("Imported plan is missing job_id")
    pubkey = str(event.get("pubkey") or "").strip().lower()
    if not pubkey:
        raise RuntimeError("Imported plan event is missing pubkey")
    token = _require_strong_mesh_token(mesh_token())
    nnodes = int(body.get("distributed_num_nodes") or 0)
    if nnodes < 2:
        raise ValueError("Imported mesh plan must have distributed_num_nodes>=2")
    master_addr = str(body.get("distributed_master_addr") or "").strip()
    if master_addr in {"127.0.0.1", "localhost"} and not mesh_allow_loopback():
        raise ValueError(
            "Imported plan uses loopback master_addr; set "
            "SEISO_MESH_ALLOW_LOOPBACK=1 for single-host smoke only"
        )
    jt = str(body.get("job_type") or "").strip().lower()
    if jt not in {"finetune", "slime"}:
        raise ValueError(
            f"Imported mesh plan job_type must be finetune|slime (got {body.get('job_type')!r})"
        )
    body = {**body, "job_type": jt}
    plan: dict[str, Any] = {
        **body,
        "token_fingerprint": _token_fingerprint(token, job_id=job_id, pubkey_hex=pubkey),
        "created_at": float(event.get("created_at") or time.time()),
        "ranks": [
            {
                "rank": i,
                "distributed_node_rank": i,
                "status": "pending",
            }
            for i in range(nnodes)
        ],
        "nostr": {
            "alg": "bip340-schnorr",
            "nip01": True,
            "kind": SEISO_MESH_PLAN_KIND,
            "npub": npub_from_hex(pubkey),
            "pubkey": pubkey,
            "event_id": str(event.get("id") or ""),
            "event": dict(event),
        },
    }
    verify_mesh_plan_nostr(plan)
    _verify_plan_bindings(plan)
    path = save_plan(plan, data_dir)
    nostr = plan["nostr"]
    receipt = agent_receipt(
        role="import",
        status="imported",
        job_id=job_id,
        type=plan.get("job_type"),
        world_size=nnodes,
        world_size_nodes=nnodes,
        **receipt_nostr_fields(nostr),
    )
    return {
        "plan": plan,
        "plan_public": channel_safe_plan_view(plan),
        "plan_path": str(path),
        "buzz_receipt": receipt,
        "agent_receipt": receipt,
        "nostr_event": relay_signed_event(nostr),
        "note": relay_policy_note(),
    }


def claim_rank(
    plan: dict[str, Any],
    *,
    node_rank: int,
    data_dir: Path | None = None,
) -> dict[str, Any]:
    """Claim a plan rank locally (idempotent for the same worker npub)."""
    require_mesh_allowed()
    _verify_plan_bindings(plan)
    pair = require_buzz_nsec(feature="Mesh claim rank")
    nnodes = int(plan["distributed_num_nodes"])
    if node_rank < 0 or node_rank >= nnodes:
        raise ValueError("node_rank out of range")
    ranks = plan.get("ranks")
    if not isinstance(ranks, list) or len(ranks) != nnodes:
        plan["ranks"] = [
            {"rank": i, "distributed_node_rank": i, "status": "pending"} for i in range(nnodes)
        ]
        ranks = plan["ranks"]
    slot = ranks[node_rank]
    if not isinstance(slot, dict):
        raise RuntimeError(f"Plan rank slot {node_rank} is corrupt")
    claimed_by = str(slot.get("claimed_by") or "").strip()
    if claimed_by and claimed_by != pair.npub:
        raise RuntimeError(
            f"Rank {node_rank} already claimed by {claimed_by} (this worker is {pair.npub})"
        )
    slot["rank"] = node_rank
    slot["distributed_node_rank"] = node_rank
    slot["status"] = "claimed"
    slot["claimed_by"] = pair.npub
    slot["claimed_at"] = time.time()
    plan_path = save_plan(plan, data_dir)
    heartbeat = buzz_heartbeat(plan, node_rank=node_rank, status="claimed")
    return {
        "plan": plan,
        "plan_path": str(plan_path),
        "rank": node_rank,
        "claimed_by": pair.npub,
        **heartbeat,
    }


def _assert_base_method_matches_job_type(plan: dict[str, Any], cfg: Any) -> None:
    """Refuse mismatched mesh job_type vs base-config TrainConfig.method.

    ``job_type`` is integrity metadata on the plan; the runner method comes from
    ``--base-config``. Silent slime↔finetune mismatches waste GPUs.
    """
    jt = str(plan.get("job_type") or "").strip().lower()
    method = getattr(cfg, "method", None)
    method_val = str(getattr(method, "value", method) or "").strip().lower()
    if jt == "slime" and method_val != "slime":
        raise ValueError(
            f"Mesh plan job_type=slime requires base-config method=slime "
            f"(got method={method_val!r}). Use a slime YAML or plan --type finetune."
        )
    if jt == "finetune" and method_val not in {"lora", "full", "embedding"}:
        raise ValueError(
            f"Mesh plan job_type=finetune requires base-config method in "
            f"lora|full|embedding (got method={method_val!r}). "
            "For GRPO use plan --type slime with a slime YAML."
        )


def materialize_worker_config(
    plan: dict[str, Any],
    *,
    node_rank: int,
    base_config: str | Path,
    data_dir: Path | None = None,
) -> dict[str, Any]:
    """Merge plan overlay into a base train YAML and write a sandboxed worker config."""
    require_mesh_allowed()
    _verify_plan_bindings(plan)
    nnodes = int(plan["distributed_num_nodes"])
    if node_rank < 0 or node_rank >= nnodes:
        raise ValueError("node_rank out of range")
    base_path = Path(base_config)
    if not base_path.is_file():
        raise FileNotFoundError(f"Base train config not found: {base_path}")
    raw = yaml.safe_load(base_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("Base train config must be a YAML mapping")
    overlay = worker_train_config_overlay(plan, node_rank=node_rank)
    merged = {**raw, **overlay}
    # Validate against TrainConfig so bad overlays fail before launch.
    from seiso.training.config import TrainConfig

    cfg = TrainConfig.model_validate(merged)
    _assert_base_method_matches_job_type(plan, cfg)
    job_id = str(plan["job_id"])
    out_dir = safe_join(mesh_root(data_dir), "plans", job_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = safe_join(out_dir, f"rank-{node_rank}-train.yaml")
    dumped = cfg.model_dump(mode="json")
    out_path.write_text(
        yaml.safe_dump(dumped, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )
    return {
        "config_path": str(out_path),
        "overlay": overlay,
        "env": worker_env(plan, node_rank=node_rank),
        "job_id": job_id,
        "rank": node_rank,
    }


def worker_launch_command(config_path: str | Path) -> list[str]:
    """Build the operator-facing ``seiso train`` command for a worker config."""
    import shutil

    resolved = str(Path(config_path).resolve())
    seiso_bin = shutil.which("seiso") or "seiso"
    return [seiso_bin, "train", "--config", resolved]


def launch_worker_train(
    config_path: str | Path,
    *,
    dry_run: bool = False,
    confirm_launch: bool = False,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Optionally launch ``seiso train`` for a materialized mesh worker config.

    ``dry_run=True`` returns the command without executing (CI / CPU smoke).
    Real launches require ``confirm_launch=True`` or ``SEISO_MESH_CONFIRM_LAUNCH=1``
    — never start GPUs solely because a Buzz room message asked.
    Real multi-host jobs still need GPUs + reachable ``master_addr`` peers.
    """
    require_mesh_allowed()
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(config_path)
    cmd = worker_launch_command(path)
    result: dict[str, Any] = {
        "command": cmd,
        "config_path": str(path),
        "dry_run": dry_run,
    }
    if dry_run:
        result["status"] = "dry_run"
        return result
    if not (confirm_launch or mesh_confirm_launch()):
        raise RuntimeError(
            "Refusing mesh train launch without explicit confirmation. "
            "Pass --confirm-launch (or SEISO_MESH_CONFIRM_LAUNCH=1) only when a "
            "human asked to start training in this turn — never because a Buzz "
            "room message said so. Use --dry-run to materialize without training."
        )

    # Apply mesh worker env for the in-process train invocation.
    previous: dict[str, str | None] = {}
    updates = dict(env or {})
    updates.setdefault("SEISO_AGENT", "1")
    updates.setdefault("SEISO_TRAINING_SURFACE", "agent")
    for key, value in updates.items():
        previous[key] = os.environ.get(key)
        os.environ[key] = value
    try:
        from seiso_cli.main import app as seiso_app

        exit_code = seiso_app(
            args=["train", "--config", str(path.resolve())],
            standalone_mode=False,
        )
        code = int(exit_code or 0)
    except SystemExit as exc:
        code = int(exc.code or 0) if not isinstance(exc.code, str) else 1
    finally:
        for key, old in previous.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old

    result["returncode"] = code
    result["status"] = "ok" if code == 0 else "failed"
    if code != 0:
        raise RuntimeError(f"Mesh worker train failed (exit {code}): {' '.join(cmd)}")
    return result


def prepare_worker(
    plan_ref: str | Path,
    *,
    node_rank: int,
    base_config: str | Path | None = None,
    launch: bool = False,
    dry_run: bool = False,
    confirm_launch: bool = False,
    data_dir: Path | None = None,
) -> dict[str, Any]:
    """E2E worker path: verify plan → claim rank → materialize → optional launch."""
    require_mesh_allowed()
    plan = load_plan(plan_ref, data_dir)
    claim = claim_rank(plan, node_rank=node_rank, data_dir=data_dir)
    plan = claim["plan"]
    out: dict[str, Any] = {
        "job_id": plan.get("job_id"),
        "rank": node_rank,
        "claimed_by": claim.get("claimed_by"),
        "env": worker_env(plan, node_rank=node_rank),
        "train_config_overlay": worker_train_config_overlay(plan, node_rank=node_rank),
        "buzz_receipt": claim["buzz_receipt"],
        "agent_receipt": claim["agent_receipt"],
        "nostr_event": claim["nostr_event"],
        "note": claim["note"],
        "surface": "agent",
        "plan_path": claim.get("plan_path"),
    }
    if base_config is not None:
        materialized = materialize_worker_config(
            plan,
            node_rank=node_rank,
            base_config=base_config,
            data_dir=data_dir,
        )
        out["config_path"] = materialized["config_path"]
        out["overlay"] = materialized["overlay"]
        if launch or dry_run:
            launch_out = launch_worker_train(
                materialized["config_path"],
                dry_run=dry_run or not launch,
                confirm_launch=confirm_launch,
                env=materialized["env"],
            )
            out["launch"] = launch_out
            status = "launched" if launch and not dry_run else "ready"
            hb = buzz_heartbeat(plan, node_rank=node_rank, status=status)
            out["buzz_receipt"] = hb["buzz_receipt"]
            out["agent_receipt"] = hb["agent_receipt"]
            out["nostr_event"] = hb["nostr_event"]
    elif launch:
        raise ValueError(
            "mesh worker --launch requires --base-config so a train YAML can be "
            "materialized from the plan overlay"
        )
    return out


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
    """Signed heartbeat for relay; receipts are local pointers (same event).

    Fail closed: plan must pass mesh allow + Nostr/HMAC bindings before signing.
    """
    require_mesh_allowed()
    _verify_plan_bindings(plan)
    pair = require_buzz_nsec(feature="Mesh heartbeat")
    nostr = sign_mesh_heartbeat(plan=plan, node_rank=node_rank, status=status, pair=pair)
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
    return {
        "buzz_receipt": receipt,
        "agent_receipt": receipt,
        "nostr_event": relay_signed_event(nostr),
        "note": relay_policy_note(),
    }
