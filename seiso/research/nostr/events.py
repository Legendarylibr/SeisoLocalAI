"""NIP-01 event serialization and BIP-340 signatures for Seiso attestations.

See https://github.com/nostr-protocol/nips/blob/master/01.md
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from seiso.research.nostr.keys import NostrKeyPair
from seiso.research.nostr.schnorr import sign_schnorr, verify_schnorr

# Addressable application kind (NIP-01: 30000 <= n < 40000) for Seiso provenance.
SEISO_PROVENANCE_KIND = 31250


def _normalize_hex(value: str, *, size: int | None = None) -> str:
    """Lowercase hex per NIP-01 wire format for id / pubkey / sig."""
    out = (value or "").strip().lower()
    if size is not None and len(out) != size:
        raise ValueError(f"expected {size}-char hex, got {len(out)}")
    int(out, 16)  # validate
    return out


def _normalize_tags(tags: Any) -> list[list[str]]:
    """NIP-01: tags are arrays of arrays of non-null strings."""
    if tags is None:
        return []
    if not isinstance(tags, list):
        raise ValueError("tags must be a list")
    normalized: list[list[str]] = []
    for tag in tags:
        if not isinstance(tag, list) or not tag:
            raise ValueError("each tag must be a non-empty list")
        normalized.append([str(part) for part in tag])
    return normalized


def canonical_event_id_preimage(
    *,
    pubkey: str,
    created_at: int,
    kind: int,
    tags: list[list[str]],
    content: str,
) -> bytes:
    # Compact UTF-8 JSON; ensure_ascii=False keeps non-ASCII content verbatim (NIP-01).
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
    pubkey = _normalize_hex(pair.public_hex, size=64)
    created_at = int(event.get("created_at") or time.time())
    kind = int(event["kind"])
    tags = _normalize_tags(event.get("tags") or [])
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
        pubkey = _normalize_hex(str(event["pubkey"]), size=64)
        created_at = int(event["created_at"])
        kind = int(event["kind"])
        tags = _normalize_tags(event.get("tags") or [])
        content = str(event.get("content") or "")
        event_id = _normalize_hex(str(event["id"]), size=64)
        sig = _normalize_hex(str(event["sig"]), size=128)
        expected = compute_event_id(
            pubkey=pubkey,
            created_at=created_at,
            kind=kind,
            tags=tags,
            content=content,
        )
        if expected != event_id:
            return False
        return verify_schnorr(bytes.fromhex(pubkey), bytes.fromhex(event_id), bytes.fromhex(sig))
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
    # Addressable: latest event per (pubkey, kind, d) is retained by relays.
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
