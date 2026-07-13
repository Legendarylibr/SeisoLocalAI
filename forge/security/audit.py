"""Structured security audit logging."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from forge.security.request_context import get_request_id

logger = logging.getLogger("seiso.audit")


def hash_audit_payload(value: Any, *, digest_chars: int = 16) -> str:
    """Stable SHA-256 prefix for audit fields (args, etc.) without logging secrets."""
    canon = json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    digest = hashlib.sha256(canon.encode("utf-8")).hexdigest()
    return digest[: max(8, digest_chars)]


def audit_event(event: str, **fields: Any) -> None:
    """Emit a structured audit record (no secrets in values)."""
    safe = {k: v for k, v in fields.items() if v is not None}
    request_id = get_request_id()
    if request_id and "request_id" not in safe:
        safe["request_id"] = request_id
    logger.info("AUDIT event=%s %s", event, json.dumps(safe, default=str))
