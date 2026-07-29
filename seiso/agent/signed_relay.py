"""Signed-relay policy for all Buzz-facing Seiso agent authority.

Policy (applies to every agent↔Buzz boundary claim):

1. **Sign locally** with the agent ``BUZZ_PRIVATE_KEY`` (NIP-01 + BIP-340).
2. **Relay only the signed event** via buzz-cli (NIP-98 to Buzz / allowlisted
   relays). Unsigned JSON is a local pointer — never channel authority.
3. **Seiso does not NIP-98 to the Buzz relay** — transport stays in buzz-cli.

This covers mesh, train/job milestones, provenance pointers, and any future
agent status that humans or peers should treat as proof.

Exceptions (must stay unsigned / local / never relayed as authority):

- Frontend Forge UI sessions (separate surface; no mesh).
- Pure local CLI work with no Buzz channel (no relay step).
- Secrets (nsec, mesh token, pay token, HF tokens) — never relay.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from seiso.agent.nostr_identity import get_buzz_keypair, require_buzz_nsec
from seiso.agent.receipts import agent_receipt, buzz_compatible_receipt
from seiso.research.nostr.events import sign_event, verify_event
from seiso.research.nostr.keys import NostrKeyPair

# Generic agent milestone / status (addressable application kind).
SEISO_AGENT_STATUS_KIND = 31254

RELAY_POLICY = "signed_event_only"

_RELAY_NOTE = (
    "Relay policy: post only the signed nostr_event (NIP-01 + BIP-340) via "
    "buzz-cli. Do not treat unsigned receipt JSON as channel authority. "
    "Applies to all Buzz↔Seiso agent authority — not mesh-only."
)


def relay_policy_note() -> str:
    return _RELAY_NOTE


def relay_signed_event(event: dict[str, Any] | None) -> dict[str, Any]:
    """Return a verified NIP-01 event or refuse (relay only with signing)."""
    if not isinstance(event, dict) or not verify_event(event):
        raise RuntimeError(
            "Refusing to relay: Nostr event missing or signature invalid. "
            "Relay only with signing."
        )
    return dict(event)


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, separators=(",", ":"), sort_keys=True, ensure_ascii=False)


def _scrub_payload(fields: dict[str, Any]) -> dict[str, Any]:
    """Drop denylisted secret keys from a payload before signing."""
    from seiso.agent.receipts import _is_forbidden_field

    return {
        key: value
        for key, value in fields.items()
        if value is not None and not _is_forbidden_field(str(key))
    }


def signed_agent_interaction(
    *,
    role: str,
    status: str,
    pair: NostrKeyPair | None = None,
    d_tag: str | None = None,
    channel: str | None = None,
    require_nsec: bool = True,
    **fields: Any,
) -> dict[str, Any]:
    """Build a Buzz-facing agent status: local receipt + relay-ready signed event.

    When ``require_nsec`` is True (default for Buzz authority), missing/invalid
    ``BUZZ_PRIVATE_KEY`` fails closed. Set False only for local-only logs.
    """
    if require_nsec:
        keypair = pair or require_buzz_nsec(feature="Agent signed status")
    else:
        keypair = pair or get_buzz_keypair()
        if keypair is None:
            receipt = agent_receipt(
                role=role,
                status=status,
                relay_policy="local_unsigned_not_authority",
                **fields,
            )
            return {
                "agent_receipt": receipt,
                "buzz_receipt": buzz_compatible_receipt(receipt),
                "nostr_event": None,
                "note": (
                    "Local unsigned receipt only — not channel authority. "
                    "Set BUZZ_PRIVATE_KEY to emit a signed nostr_event for Buzz."
                ),
            }

    payload = _scrub_payload(
        {
            "role": role,
            "status": status,
            **fields,
        }
    )
    # Ensure role/status are inside signed content even after scrub envelope strip.
    payload["role"] = role
    payload["status"] = status
    if channel is not None:
        payload["channel"] = channel

    d_value = d_tag or (
        f"{fields.get('job_id') or uuid.uuid4().hex}:{role}:{status}"
    )
    tags = [
        ["d", d_value],
        ["t", "seiso-agent-status"],
        ["client", "seiso"],
        ["role", role],
    ]
    if channel:
        tags.append(["channel", str(channel)])

    draft = {
        "kind": SEISO_AGENT_STATUS_KIND,
        "created_at": int(time.time()),
        "tags": tags,
        "content": _canonical_json(payload),
    }
    event = sign_event(draft, keypair)
    receipt = agent_receipt(
        role=role,
        status=status,
        npub=keypair.npub,
        nostr_event_id=event["id"],
        nostr_kind=SEISO_AGENT_STATUS_KIND,
        sig_alg="bip340-schnorr",
        nip01=True,
        relay_policy=RELAY_POLICY,
        channel=channel,
        **fields,
    )
    return {
        "agent_receipt": receipt,
        "buzz_receipt": buzz_compatible_receipt(receipt),
        "nostr_event": relay_signed_event(event),
        "note": relay_policy_note(),
    }
