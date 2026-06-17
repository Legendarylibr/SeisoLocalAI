"""Structured security audit logging."""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger("seiso.audit")


def audit_event(event: str, **fields: Any) -> None:
    """Emit a structured audit record (no secrets in values)."""
    safe = {k: v for k, v in fields.items() if v is not None}
    logger.info("AUDIT event=%s %s", event, json.dumps(safe, default=str))
