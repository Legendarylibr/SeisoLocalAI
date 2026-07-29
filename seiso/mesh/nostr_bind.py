"""Nostr (NIP-01 + BIP-340) binding for mesh plans and receipts.

Plans are signed as addressable events (kind 31251). Workers verify the
Schnorr signature and that the event content matches the local plan body.
Optional ``SEISO_MESH_TRUSTED_NPUBS`` / ``SEISO_MESH_TRUSTED_PUBKEYS`` allowlists
restrict which agent keys may author plans.

**Relay policy:** publish only the signed NIP-01 ``event`` (via buzz-cli /
allowlisted relays). Unsigned receipt JSON is local metadata — not authority
and not what peers should trust from a channel. Seiso itself does not NIP-98
to the Buzz relay.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from seiso.research.nostr.bech32 import bech32_decode
from seiso.research.nostr.events import sign_event, verify_event
from seiso.research.nostr.keys import NostrKeyPair, npub_from_hex

# Addressable application kinds (NIP-01: 30000 <= n < 40000).
SEISO_MESH_PLAN_KIND = 31251
SEISO_MESH_ANNOUNCE_KIND = 31252
SEISO_MESH_HEARTBEAT_KIND = 31253

_RELAY_ONLY_WITH_SIGNING_NOTE = (
    "Relay policy: post only the signed nostr_event (NIP-01 + BIP-340) via "
    "buzz-cli. Do not treat unsigned receipt JSON as channel authority."
)

_SIGNED_PLAN_KEYS = (
    "job_id",
    "channel",
    "job_type",
    "preset",
    "world_size_nodes",
    "distributed_num_nodes",
    "distributed_nproc_per_node",
    "distributed_master_addr",
    "distributed_master_port",
    "distributed_strategy",
    "multi_gpu",
    "protocol_fee_sats",
    "market",
    # token_fingerprint stays local-only (HMAC check) — never in signed/relayed content.
)


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, separators=(",", ":"), sort_keys=True, ensure_ascii=False)


def plan_signed_payload(plan: dict[str, Any]) -> dict[str, Any]:
    return {key: plan[key] for key in _SIGNED_PLAN_KEYS if key in plan}


def _trusted_pubkey_hexes() -> set[str]:
    allowed: set[str] = set()
    for env_name in ("SEISO_MESH_TRUSTED_PUBKEYS", "SEISO_MESH_TRUSTED_NPUBS"):
        raw = (os.environ.get(env_name) or "").strip()
        if not raw:
            continue
        for part in raw.split(","):
            item = part.strip()
            if not item:
                continue
            if item.startswith("npub1"):
                hrp, data = bech32_decode(item)
                if hrp != "npub" or len(data) != 32:
                    raise RuntimeError(f"invalid npub in {env_name}: {item[:12]}…")
                allowed.add(data.hex())
            else:
                hexed = item.lower()
                if len(hexed) != 64:
                    raise RuntimeError(f"invalid pubkey in {env_name}")
                int(hexed, 16)
                allowed.add(hexed)
    return allowed


def sign_mesh_plan(plan: dict[str, Any], pair: NostrKeyPair) -> dict[str, Any]:
    """Attach a NIP-01 signed event covering the plan body."""
    payload = plan_signed_payload(plan)
    content = _canonical_json(payload)
    draft = {
        "kind": SEISO_MESH_PLAN_KIND,
        "created_at": int(plan.get("created_at") or time.time()),
        "tags": [
            ["d", str(plan["job_id"])],
            ["t", "seiso-mesh-plan"],
            ["client", "seiso"],
            ["channel", str(plan.get("channel") or "")],
        ],
        "content": content,
    }
    event = sign_event(draft, pair)
    plan["nostr"] = {
        "alg": "bip340-schnorr",
        "nip01": True,
        "kind": SEISO_MESH_PLAN_KIND,
        "npub": pair.npub,
        "pubkey": pair.public_hex,
        "event_id": event["id"],
        "event": event,
    }
    return plan


def verify_mesh_plan_nostr(plan: dict[str, Any]) -> None:
    """Fail closed unless the plan carries a valid Nostr signature matching its body."""
    nostr = plan.get("nostr")
    if not isinstance(nostr, dict):
        raise RuntimeError(
            "Mesh plan is missing Nostr signature (nostr). "
            "Recreate with seiso mesh plan under a valid BUZZ_PRIVATE_KEY nsec."
        )
    event = nostr.get("event")
    if not isinstance(event, dict) or not verify_event(event):
        raise RuntimeError("Mesh plan Nostr event signature is invalid")
    if int(event.get("kind") or 0) != SEISO_MESH_PLAN_KIND:
        raise RuntimeError("Mesh plan Nostr event has unexpected kind")
    try:
        body = json.loads(str(event.get("content") or ""))
    except json.JSONDecodeError as exc:
        raise RuntimeError("Mesh plan Nostr event content is not JSON") from exc
    expected = plan_signed_payload(plan)
    if body != expected:
        raise RuntimeError(
            "Mesh plan body does not match Nostr-signed content (tamper detected)"
        )
    pubkey = str(event.get("pubkey") or "").lower()
    if pubkey != str(nostr.get("pubkey") or "").lower():
        raise RuntimeError("Mesh plan nostr.pubkey does not match event.pubkey")
    event_id = str(nostr.get("event_id") or "").strip().lower()
    signed_id = str(event.get("id") or "").strip().lower()
    if not event_id:
        raise RuntimeError("Mesh plan is missing nostr.event_id")
    if event_id != signed_id:
        raise RuntimeError("Mesh plan nostr.event_id does not match signed event id")
    trusted = _trusted_pubkey_hexes()
    if trusted and pubkey not in trusted:
        raise RuntimeError(
            "Mesh plan author pubkey is not in SEISO_MESH_TRUSTED_NPUBS/"
            "SEISO_MESH_TRUSTED_PUBKEYS"
        )


def sign_mesh_announce(
    record: dict[str, Any], pair: NostrKeyPair
) -> dict[str, Any]:
    """Sign an announce record (local + receipt metadata)."""
    payload = {
        "channel": record.get("channel"),
        "gpus": record.get("gpus"),
        "capabilities": record.get("capabilities"),
        "alias": record.get("alias"),
        "mesh_endpoint_fingerprint": record.get("mesh_endpoint_fingerprint"),
    }
    draft = {
        "kind": SEISO_MESH_ANNOUNCE_KIND,
        "created_at": int(record.get("ts") or time.time()),
        "tags": [
            ["d", str(record.get("mesh_endpoint_fingerprint") or "")],
            ["t", "seiso-mesh-announce"],
            ["client", "seiso"],
            ["channel", str(record.get("channel") or "")],
        ],
        "content": _canonical_json(payload),
    }
    event = sign_event(draft, pair)
    return {
        "alg": "bip340-schnorr",
        "nip01": True,
        "kind": SEISO_MESH_ANNOUNCE_KIND,
        "npub": pair.npub,
        "pubkey": pair.public_hex,
        "event_id": event["id"],
        "event": event,
    }


def sign_mesh_heartbeat(
    *,
    plan: dict[str, Any],
    node_rank: int,
    status: str,
    pair: NostrKeyPair,
) -> dict[str, Any]:
    """Sign a worker heartbeat bound to the plan job id."""
    job_id = str(plan.get("job_id") or "")
    payload = {
        "job_id": job_id,
        "type": plan.get("job_type"),
        "rank": int(node_rank),
        "status": status,
        "plan_event_id": (plan.get("nostr") or {}).get("event_id"),
    }
    draft = {
        "kind": SEISO_MESH_HEARTBEAT_KIND,
        "created_at": int(time.time()),
        "tags": [
            ["d", f"{job_id}:{node_rank}:{status}"],
            ["t", "seiso-mesh-heartbeat"],
            ["client", "seiso"],
            ["e", str((plan.get("nostr") or {}).get("event_id") or "")],
        ],
        "content": _canonical_json(payload),
    }
    event = sign_event(draft, pair)
    return {
        "alg": "bip340-schnorr",
        "nip01": True,
        "kind": SEISO_MESH_HEARTBEAT_KIND,
        "npub": pair.npub,
        "pubkey": pair.public_hex,
        "event_id": event["id"],
        "event": event,
    }


def receipt_nostr_fields(nostr: dict[str, Any]) -> dict[str, Any]:
    """Local/agent receipt pointers (not a substitute for the signed event)."""
    return {
        "npub": nostr.get("npub") or npub_from_hex(str(nostr.get("pubkey") or "")),
        "nostr_event_id": nostr.get("event_id"),
        "nostr_kind": nostr.get("kind"),
        "sig_alg": nostr.get("alg") or "bip340-schnorr",
        "nip01": True,
        "relay_policy": "signed_event_only",
    }


def relay_signed_event(nostr: dict[str, Any]) -> dict[str, Any]:
    """Return the only channel/relay-worthy payload: the verified NIP-01 event."""
    event = nostr.get("event")
    if not isinstance(event, dict) or not verify_event(event):
        raise RuntimeError(
            "Refusing to relay: mesh Nostr event missing or signature invalid. "
            "Relay only with signing."
        )
    return dict(event)


def relay_policy_note() -> str:
    return _RELAY_ONLY_WITH_SIGNING_NOTE
