"""Minimal NIP-01 WebSocket publish / fetch client."""

from __future__ import annotations

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
    event_id = str(event.get("id") or "")
    for raw in relays:
        url = validate_relay_url(
            raw, allowlist=allowlist, allow_loopback=allow_loopback
        )
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
                    if msg[0] == "OK" and len(msg) >= 3 and msg[1] == event_id:
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
    eid = (event_id or "").strip().lower()
    if len(eid) != 64:
        raise ValueError("event_id must be 64-char hex")
    for raw in relays:
        url = validate_relay_url(
            raw, allowlist=allowlist, allow_loopback=allow_loopback
        )
        sub_id = uuid.uuid4().hex[:16]
        try:
            with connect(url, open_timeout=timeout_s, close_timeout=5) as ws:
                ws.send(json.dumps(["REQ", sub_id, {"ids": [eid], "limit": 1}]))
                for _ in range(12):
                    raw_msg = ws.recv(timeout=timeout_s)
                    msg = json.loads(raw_msg)
                    if not isinstance(msg, list) or not msg:
                        continue
                    if msg[0] == "EVENT" and len(msg) >= 3 and msg[1] == sub_id:
                        event = msg[2]
                        if isinstance(event, dict) and str(event.get("id")) == eid:
                            try:
                                ws.send(json.dumps(["CLOSE", sub_id]))
                            except Exception:
                                pass
                            return event
                    if msg[0] == "EOSE" and len(msg) >= 2 and msg[1] == sub_id:
                        break
        except Exception as exc:
            logger.warning("relay fetch failed for %s: %s", url, exc)
    return None
