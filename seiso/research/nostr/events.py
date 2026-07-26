"""NIP-01 event serialization and BIP-340 signatures for Seiso attestations."""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from seiso.research.nostr.keys import NostrKeyPair
from seiso.research.nostr.schnorr import sign_schnorr, verify_schnorr

# Parameterized replaceable application kind (Seiso provenance).
SEISO_PROVENANCE_KIND = 31250


def canonical_event_id_preimage(
    *,
    pubkey: str,
    created_at: int,
    kind: int,
    tags: list[list[str]],
    content: str,
) -> bytes:
    payload = [0, pubkey, created_at, kind, tags, content]
    serialized = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    return serialized.encode("utf-8")


def compute_event_id(
    *,
    pubkey: str,
    created_at: int,
    kind: int,
    tags: list[list[str]],
    content: str,
) -> str:
    return hashlib.sha256(
        canonical_event_id_preimage(
            pubkey=pubkey,
            created_at=created_at,
            kind=kind,
            tags=tags,
            content=content,
        )
    ).hexdigest()


def sign_event(event: dict[str, Any], pair: NostrKeyPair) -> dict[str, Any]:
    pubkey = pair.public_hex
    created_at = int(event.get("created_at") or time.time())
    kind = int(event["kind"])
    tags = list(event.get("tags") or [])
    content = str(event.get("content") or "")
    event_id = compute_event_id(
        pubkey=pubkey,
        created_at=created_at,
        kind=kind,
        tags=tags,
        content=content,
    )
    sig = sign_schnorr(bytes.fromhex(pair.secret_hex), bytes.fromhex(event_id)).hex()
    return {
        "id": event_id,
        "pubkey": pubkey,
        "created_at": created_at,
        "kind": kind,
        "tags": tags,
        "content": content,
        "sig": sig,
    }


def verify_event(event: dict[str, Any]) -> bool:
    try:
        pubkey = str(event["pubkey"])
        created_at = int(event["created_at"])
        kind = int(event["kind"])
        tags = list(event.get("tags") or [])
        content = str(event.get("content") or "")
        event_id = str(event["id"])
        sig = str(event["sig"])
        expected = compute_event_id(
            pubkey=pubkey,
            created_at=created_at,
            kind=kind,
            tags=tags,
            content=content,
        )
        if expected != event_id:
            return False
        return verify_schnorr(
            bytes.fromhex(pubkey), bytes.fromhex(event_id), bytes.fromhex(sig)
        )
    except Exception:
        return False


def build_attestation_event(
    *,
    pair: NostrKeyPair,
    attestation_json: str,
    pipeline: str,
    run_id: str,
    created_at: int | None = None,
) -> dict[str, Any]:
    tags = [
        ["d", f"{pipeline}:{run_id}"],
        ["t", "seiso-provenance"],
        ["client", "seiso"],
    ]
    draft = {
        "kind": SEISO_PROVENANCE_KIND,
        "created_at": int(created_at if created_at is not None else time.time()),
        "tags": tags,
        "content": attestation_json,
    }
    return sign_event(draft, pair)
