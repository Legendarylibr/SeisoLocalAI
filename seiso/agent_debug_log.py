"""Temporary agent debug logging for native Linux inference investigation."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

_SESSION_ID = "15b90d"


def agent_debug_enabled() -> bool:
    return os.environ.get("SEISO_AGENT_DEBUG", "1").strip().lower() not in {
        "0",
        "false",
        "no",
    }


def _log_path() -> Path:
    override = os.environ.get("SEISO_DEBUG_LOG", "").strip()
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[1] / ".cursor" / f"debug-{_SESSION_ID}.log"


def agent_debug_log(
    *,
    hypothesis_id: str,
    location: str,
    message: str,
    data: dict[str, Any] | None = None,
    run_id: str | None = None,
) -> None:
    if not agent_debug_enabled():
        return
    payload = {
        "sessionId": _SESSION_ID,
        "runId": run_id or os.environ.get("SEISO_AGENT_DEBUG_RUN", "pre-fix"),
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data or {},
        "timestamp": int(time.time() * 1000),
    }
    path = _log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=str) + "\n")
