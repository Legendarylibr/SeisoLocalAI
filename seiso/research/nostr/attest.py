"""Seal local manifests to Nostr and verify external commitments."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from seiso.research.nostr.events import (
    SEISO_PROVENANCE_KIND,
    build_attestation_event,
    verify_event,
)
from seiso.research.nostr.keys import NostrKeyPair, load_keypair
from seiso.research.nostr.policy import (
    normalize_relay_list,
    nostr_allowed,
    nostr_auto_attest_enabled,
    relay_allowlist_from_env,
)
from seiso.research.nostr.relays import fetch_event_by_id, publish_event
from seiso.research.provenance import (
    ATTESTATION_SCHEMA_V1,
    ATTESTATION_SCHEMA_V2,
    attestation_content_json,
    build_attestation_v1,
    content_fingerprint,
    merge_nostr_receipt,
    write_manifest_with_nostr_receipt,
)
from seiso.security import SecurityError

logger = logging.getLogger(__name__)


def _load_manifest(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"manifest root must be an object: {path}")
    return data


def attest_manifest(
    path: Path | str,
    *,
    relays: list[str] | None = None,
    pair: NostrKeyPair | None = None,
    identity: str = "cli",
    data_dir: Path | None = None,
    allowlist: list[str] | None = None,
    allow_loopback: bool = False,
    require_allow: bool = True,
) -> dict[str, Any]:
    """Build attestation, publish to relays, write receipt into the manifest."""
    if require_allow and not nostr_allowed():
        raise SecurityError("Nostr disabled; set SEISO_ALLOW_NOSTR=1 to enable")

    manifest_path = Path(path)
    manifest = _load_manifest(manifest_path)
    key = pair or load_keypair(identity=identity, data_dir=data_dir)
    if key is None:
        raise SecurityError(
            f"No Nostr key for identity {identity!r}; run `seiso provenance keygen`"
        )

    relay_urls = list(relays or relay_allowlist_from_env())
    host_allowlist = allowlist
    if host_allowlist is None and relay_urls:
        # Default allowlist = hostnames of the configured relay URLs.
        from urllib.parse import urlparse

        host_allowlist = [
            (urlparse(r).hostname or "").lower() for r in relay_urls if r.strip()
        ]
    normalized = normalize_relay_list(
        relay_urls, allowlist=host_allowlist, allow_loopback=allow_loopback
    )

    attestation = build_attestation_v1(manifest, manifest_path=manifest_path)
    content = attestation_content_json(attestation)
    event = build_attestation_event(
        pair=key,
        attestation_json=content,
        pipeline=str(attestation["pipeline"]),
        run_id=str(attestation["run_id"]),
    )
    accepted = publish_event(
        event,
        normalized,
        allowlist=host_allowlist,
        allow_loopback=allow_loopback,
    )
    receipt = {
        "event_id": event["id"],
        "pubkey": event["pubkey"],
        "relays": accepted,
        "published_at": datetime.now(timezone.utc).isoformat(),
        "attestation_sha256": content_fingerprint(attestation),
        "kind": event["kind"],
        "d_tag": f"{attestation['pipeline']}:{attestation['run_id']}",
    }
    updated = merge_nostr_receipt(manifest, receipt)
    write_manifest_with_nostr_receipt(manifest_path, updated)
    return {
        "ok": True,
        "manifest_path": str(manifest_path),
        "attestation": attestation,
        "receipt": receipt,
        "event_id": event["id"],
    }


def verify_attestation(
    path: Path | str,
    *,
    relays: list[str] | None = None,
    allowlist: list[str] | None = None,
    allow_loopback: bool = False,
    require_network: bool = True,
) -> dict[str, Any]:
    """Recompute local attestation digests and optionally fetch/verify the event."""
    manifest_path = Path(path)
    manifest = _load_manifest(manifest_path)
    receipt = manifest.get("nostr")
    if not isinstance(receipt, dict) or not receipt.get("event_id"):
        return {
            "ok": False,
            "error": "manifest has no nostr receipt; run `seiso provenance attest` first",
            "manifest_path": str(manifest_path),
        }

    attestation = build_attestation_v1(manifest, manifest_path=manifest_path)
    local_attestation_sha = content_fingerprint(attestation)
    stored_attestation_sha = str(receipt.get("attestation_sha256") or "")
    local_ok = bool(stored_attestation_sha) and local_attestation_sha == stored_attestation_sha

    report: dict[str, Any] = {
        "ok": local_ok,
        "manifest_path": str(manifest_path),
        "local_attestation_match": local_ok,
        "attestation": attestation,
        "receipt": receipt,
        "event_verified": None,
        "event": None,
    }
    if not require_network:
        return report

    if not nostr_allowed():
        report["ok"] = False
        report["error"] = "Nostr disabled; set SEISO_ALLOW_NOSTR=1 to fetch events"
        return report

    relay_urls = list(relays or receipt.get("relays") or relay_allowlist_from_env())
    if not relay_urls:
        report["ok"] = False
        report["error"] = "no relays configured for verification"
        return report

    from urllib.parse import urlparse

    host_allowlist = allowlist
    if host_allowlist is None:
        host_allowlist = [
            (urlparse(r).hostname or "").lower() for r in relay_urls if r.strip()
        ]
    normalized = normalize_relay_list(
        relay_urls, allowlist=host_allowlist, allow_loopback=allow_loopback
    )
    event = fetch_event_by_id(
        str(receipt["event_id"]),
        normalized,
        allowlist=host_allowlist,
        allow_loopback=allow_loopback,
    )
    if event is None:
        report["ok"] = False
        report["error"] = "event not found on configured relays"
        return report

    sig_ok = verify_event(event)
    content = str(event.get("content") or "")
    try:
        remote_attestation = json.loads(content)
    except json.JSONDecodeError:
        remote_attestation = None
    remote_schema = (
        remote_attestation.get("schema") if isinstance(remote_attestation, dict) else None
    )
    remote_match = (
        isinstance(remote_attestation, dict)
        and remote_schema in {ATTESTATION_SCHEMA_V1, ATTESTATION_SCHEMA_V2}
        and content_fingerprint(remote_attestation) == local_attestation_sha
    )
    # Also accept exact content string match against canonical local JSON.
    if not remote_match:
        remote_match = content == attestation_content_json(attestation)

    receipt_pubkey = str(receipt.get("pubkey") or "").strip().lower()
    event_pubkey = str(event.get("pubkey") or "").strip().lower()
    pubkey_ok = bool(receipt_pubkey) and receipt_pubkey == event_pubkey
    kind_raw = event.get("kind")
    try:
        kind_ok = kind_raw is not None and int(kind_raw) == SEISO_PROVENANCE_KIND
    except (TypeError, ValueError):
        kind_ok = False
    expected_d = str(
        receipt.get("d_tag")
        or f"{attestation.get('pipeline')}:{attestation.get('run_id')}"
    )
    tags = event.get("tags") or []
    d_tags = [
        str(t[1])
        for t in tags
        if isinstance(t, list) and len(t) >= 2 and str(t[0]) == "d"
    ]
    d_ok = bool(d_tags) and expected_d in d_tags

    report["event"] = {
        "id": event.get("id"),
        "pubkey": event.get("pubkey"),
        "created_at": event.get("created_at"),
        "kind": event.get("kind"),
    }
    report["event_pubkey_match"] = pubkey_ok
    report["event_kind_ok"] = kind_ok
    report["event_d_tag_ok"] = d_ok
    report["event_verified"] = bool(
        sig_ok and remote_match and pubkey_ok and kind_ok and d_ok
    )
    report["ok"] = bool(local_ok and report["event_verified"])
    if not report["ok"] and not report.get("error"):
        report["error"] = "attestation mismatch or invalid event signature"
    return report


def maybe_auto_attest(
    path: Path | str,
    *,
    identity: str = "cli",
    data_dir: Path | None = None,
    relays: list[str] | None = None,
    allow_loopback: bool = False,
) -> dict[str, Any] | None:
    """Non-fatal auto-attest when SEISO_ALLOW_NOSTR + SEISO_NOSTR_ATTEST are set."""
    if not nostr_auto_attest_enabled():
        return None
    try:
        return attest_manifest(
            path,
            identity=identity,
            data_dir=data_dir,
            relays=relays,
            allow_loopback=allow_loopback,
            require_allow=True,
        )
    except Exception as exc:
        logger.warning("Nostr auto-attest skipped for %s: %s", path, exc)
        return {"ok": False, "error": str(exc), "manifest_path": str(path)}
