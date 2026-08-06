"""Minimal NIP-01 WebSocket publish / fetch client.

Wire messages follow https://github.com/nostr-protocol/nips/blob/master/01.md
(EVENT / REQ / CLOSE client→relay; EVENT / OK / EOSE / CLOSED / NOTICE relay→client).
"""

from __future__ import annotations

import contextlib
import json
import logging
import uuid
from typing import Any

from seiso.research.nostr.policy import validate_relay_url
from seiso.security import SecurityError

logger = logging.getLogger(__name__)


def _require_websockets():
    try:
        from websockets.sync.client import connect
    except ImportError as exc:
        raise ImportError(
            "Nostr relay I/O requires optional deps: pip install 'seiso[nostr]'"
        ) from exc
    return connect


def _hex64(value: str, *, label: str) -> str:
    out = (value or "").strip().lower()
    if len(out) != 64:
        raise ValueError(f"{label} must be 64-char hex")
    int(out, 16)
    return out


def _read_subscription_event(
    ws: Any,
    *,
    sub_id: str,
    timeout_s: float,
    match: Any,
    max_reads: int = 12,
) -> dict[str, Any] | None:
    """Read until a matching EVENT, EOSE, or CLOSED for ``sub_id``."""
    for _ in range(max_reads):
        raw_msg = ws.recv(timeout=timeout_s)
        msg = json.loads(raw_msg)
        if not isinstance(msg, list) or not msg:
            continue
        typ = msg[0]
        if typ == "EVENT" and len(msg) >= 3 and msg[1] == sub_id:
            event = msg[2]
            if isinstance(event, dict) and match(event):
                with contextlib.suppress(Exception):
                    ws.send(json.dumps(["CLOSE", sub_id]))
                return event
        if typ == "EOSE" and len(msg) >= 2 and msg[1] == sub_id:
            break
        if typ == "CLOSED" and len(msg) >= 2 and msg[1] == sub_id:
            logger.debug("relay closed subscription %s: %s", sub_id, msg[2:])
            break
        if typ == "NOTICE":
            logger.debug("relay notice: %s", msg[1:])
    return None


def publish_event(
    event: dict[str, Any],
    relays: list[str],
    *,
    allowlist: list[str] | None = None,
    allow_loopback: bool = False,
    timeout_s: float = 15.0,
) -> list[str]:
    """Publish event to relays; return list of relays that acknowledged OK."""
    connect = _require_websockets()
    accepted: list[str] = []
    event_id = str(event.get("id") or "").strip().lower()
    for raw in relays:
        url = validate_relay_url(raw, allowlist=allowlist, allow_loopback=allow_loopback)
        try:
            with connect(url, open_timeout=timeout_s, close_timeout=5) as ws:
                ws.send(json.dumps(["EVENT", event]))
                deadline_reads = 8
                ok = False
                for _ in range(deadline_reads):
                    raw_msg = ws.recv(timeout=timeout_s)
                    msg = json.loads(raw_msg)
                    if not isinstance(msg, list) or not msg:
                        continue
                    if msg[0] == "OK" and len(msg) >= 3 and str(msg[1]).strip().lower() == event_id:
                        ok = bool(msg[2])
                        break
                    if msg[0] == "NOTICE":
                        logger.debug("relay notice from %s: %s", url, msg[1:])
                if ok:
                    accepted.append(url)
                else:
                    logger.warning("relay %s did not OK event %s", url, event_id[:16])
        except Exception as exc:
            logger.warning("relay publish failed for %s: %s", url, exc)
    if not accepted:
        raise SecurityError("no allowlisted relay accepted the attestation event")
    return accepted


def fetch_event_by_id(
    event_id: str,
    relays: list[str],
    *,
    allowlist: list[str] | None = None,
    allow_loopback: bool = False,
    timeout_s: float = 15.0,
) -> dict[str, Any] | None:
    """Fetch a single event by id from the first relay that returns it."""
    connect = _require_websockets()
    eid = _hex64(event_id, label="event_id")
    for raw in relays:
        url = validate_relay_url(raw, allowlist=allowlist, allow_loopback=allow_loopback)
        sub_id = uuid.uuid4().hex[:16]
        try:
            with connect(url, open_timeout=timeout_s, close_timeout=5) as ws:
                ws.send(json.dumps(["REQ", sub_id, {"ids": [eid], "limit": 1}]))
                found = _read_subscription_event(
                    ws,
                    sub_id=sub_id,
                    timeout_s=timeout_s,
                    match=lambda ev: str(ev.get("id") or "").strip().lower() == eid,
                )
                if found is not None:
                    return found
        except Exception as exc:
            logger.warning("relay fetch failed for %s: %s", url, exc)
    return None


def fetch_addressable_event(
    *,
    pubkey: str,
    kind: int,
    d_tag: str,
    relays: list[str],
    allowlist: list[str] | None = None,
    allow_loopback: bool = False,
    timeout_s: float = 15.0,
) -> dict[str, Any] | None:
    """Fetch latest addressable event for (authors, kinds, #d) per NIP-01."""
    connect = _require_websockets()
    author = _hex64(pubkey, label="pubkey")
    d_value = str(d_tag or "").strip()
    if not d_value:
        raise ValueError("d_tag is required")
    filt = {
        "authors": [author],
        "kinds": [int(kind)],
        "#d": [d_value],
        "limit": 1,
    }
    for raw in relays:
        url = validate_relay_url(raw, allowlist=allowlist, allow_loopback=allow_loopback)
        sub_id = uuid.uuid4().hex[:16]
        try:
            with connect(url, open_timeout=timeout_s, close_timeout=5) as ws:
                ws.send(json.dumps(["REQ", sub_id, filt]))
                found = _read_subscription_event(
                    ws,
                    sub_id=sub_id,
                    timeout_s=timeout_s,
                    match=lambda ev: (
                        str(ev.get("pubkey") or "").strip().lower() == author
                        and int(ev.get("kind")) == int(kind)
                    ),
                )
                if found is not None:
                    return found
        except Exception as exc:
            logger.warning("relay addressable fetch failed for %s: %s", url, exc)
    return None
