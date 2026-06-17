"""Optional AutoDefense integration — scan LLM inputs/outputs for prompt injection."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

import httpx

from forge.config import ForgeSettings, get_settings
from forge.security.audit import audit_event
from seiso.security import SecurityError

logger = logging.getLogger(__name__)

_BLOCK_ACTIONS = frozenset({"block", "block_isolate", "block_output"})
_LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


class DefenseBlockedError(Exception):
    """Raised when AutoDefense blocks an interaction."""

    def __init__(self, message: str, *, result: DefenseResult) -> None:
        super().__init__(message)
        self.result = result


@dataclass
class DefenseResult:
    session_id: str | None = None
    trace_id: str | None = None
    risk_score: int = 0
    action: str = "allow"
    sanitized_input: str | None = None
    sanitized_output: str | None = None
    threat_types: list[str] = field(default_factory=list)
    top_reasons: list[str] = field(default_factory=list)
    blocked: bool = False
    unavailable: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "trace_id": self.trace_id,
            "risk_score": self.risk_score,
            "action": self.action,
            "threat_types": self.threat_types,
            "top_reasons": self.top_reasons[:5],
            "blocked": self.blocked,
            "unavailable": self.unavailable,
        }


def validate_autodefense_url(url: str) -> str:
    """Allow only local AutoDefense endpoints."""
    raw = (url or "").strip().rstrip("/")
    if not raw:
        raise SecurityError("AutoDefense URL is required when enabled")

    parsed = urlparse(raw)
    host = (parsed.hostname or "").lower().rstrip(".")
    if parsed.scheme != "http" or host not in _LOCAL_HOSTS:
        raise SecurityError("AutoDefense URL must be http://127.0.0.1 or http://localhost")
    port = parsed.port or 80
    if port < 1 or port > 65535:
        raise SecurityError("AutoDefense URL has invalid port")
    return f"http://{host}:{port}"


def _headers(settings: ForgeSettings) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if settings.autodefense_api_key:
        headers["Authorization"] = f"Bearer {settings.autodefense_api_key}"
    return headers


def _parse_response(data: dict[str, Any]) -> DefenseResult:
    explain = data.get("explain") or {}
    action = str(data.get("action") or "allow")
    return DefenseResult(
        session_id=data.get("session_id"),
        trace_id=data.get("trace_id"),
        risk_score=int(data.get("risk_score") or 0),
        action=action,
        sanitized_input=data.get("sanitized_input"),
        sanitized_output=data.get("sanitized_output"),
        threat_types=list(explain.get("threat_types") or []),
        top_reasons=list(explain.get("top_reasons") or [])[:5],
        blocked=action in _BLOCK_ACTIONS,
    )


def defense_enabled(settings: ForgeSettings | None = None, *, request_flag: bool | None = None) -> bool:
    """Resolve whether defense is active for this request."""
    cfg = settings or get_settings()
    if not cfg.autodefense_enabled:
        return False
    if request_flag is None:
        return True
    return request_flag


async def check_health(settings: ForgeSettings | None = None) -> dict[str, Any]:
    """Probe AutoDefense /health endpoint."""
    cfg = settings or get_settings()
    if not cfg.autodefense_enabled:
        return {"enabled": False, "reachable": False, "status": "disabled"}

    try:
        base = validate_autodefense_url(cfg.autodefense_url)
    except SecurityError as exc:
        return {"enabled": True, "reachable": False, "status": "invalid_url", "error": str(exc)}

    try:
        async with httpx.AsyncClient(timeout=cfg.autodefense_timeout) as client:
            res = await client.get(f"{base}/health", headers=_headers(cfg))
            if res.status_code == 200:
                body = res.json()
                return {
                    "enabled": True,
                    "reachable": True,
                    "status": body.get("status", "ok"),
                    "url": base,
                }
            return {"enabled": True, "reachable": False, "status": f"http_{res.status_code}", "url": base}
    except httpx.HTTPError as exc:
        return {"enabled": True, "reachable": False, "status": "unreachable", "error": str(exc), "url": base}


async def analyze(
    user_input: str,
    *,
    model_output: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    session_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    settings: ForgeSettings | None = None,
) -> DefenseResult:
    """Call AutoDefense POST /analyze."""
    cfg = settings or get_settings()
    base = validate_autodefense_url(cfg.autodefense_url)

    payload: dict[str, Any] = {"user_input": user_input[:50_000]}
    if model_output is not None:
        payload["model_output"] = model_output[:100_000]
    if tool_calls:
        payload["tool_calls"] = tool_calls
    if session_id:
        payload["session_id"] = session_id
    if metadata:
        payload["metadata"] = metadata

    try:
        async with httpx.AsyncClient(timeout=cfg.autodefense_timeout) as client:
            res = await client.post(f"{base}/analyze", headers=_headers(cfg), json=payload)
            if res.status_code >= 400:
                logger.warning("AutoDefense analyze failed: HTTP %s", res.status_code)
                if cfg.autodefense_fail_open:
                    return DefenseResult(action="allow", unavailable=True)
                raise DefenseBlockedError(
                    f"AutoDefense returned HTTP {res.status_code}",
                    result=DefenseResult(action="block", blocked=True, unavailable=True),
                )
            return _parse_response(res.json())
    except httpx.HTTPError as exc:
        logger.warning("AutoDefense unreachable: %s", exc)
        audit_event("autodefense_unavailable", error=str(exc))
        if cfg.autodefense_fail_open:
            return DefenseResult(action="allow", unavailable=True)
        raise DefenseBlockedError(
            "AutoDefense is unreachable",
            result=DefenseResult(action="block", blocked=True, unavailable=True),
        ) from exc


def extract_user_input(messages: list[dict[str, Any]]) -> str:
    """Concatenate user messages for defense scanning."""
    parts = [str(m.get("content") or "") for m in messages if m.get("role") == "user"]
    return "\n\n".join(p for p in parts if p.strip())


def apply_input_sanitization(messages: list[dict[str, Any]], sanitized: str) -> list[dict[str, Any]]:
    """Replace the last user message content with sanitized input."""
    if not messages:
        return messages
    out = [dict(m) for m in messages]
    for i in range(len(out) - 1, -1, -1):
        if out[i].get("role") == "user":
            out[i] = {**out[i], "content": sanitized}
            break
    return out


def enforce_input(result: DefenseResult, *, block_message: str | None = None) -> str | None:
    """Return sanitized input to use, or raise if blocked."""
    if result.blocked:
        msg = block_message or "Input blocked by AutoDefense"
        if result.top_reasons:
            msg = f"{msg}: {result.top_reasons[0]}"
        raise DefenseBlockedError(msg, result=result)
    if result.action == "sanitize" and result.sanitized_input is not None:
        return result.sanitized_input
    return None


def enforce_output(result: DefenseResult, reply: str, *, block_message: str | None = None) -> str:
    """Return sanitized output or raise if blocked."""
    if result.blocked:
        msg = block_message or "Output blocked by AutoDefense"
        if result.top_reasons:
            msg = f"{msg}: {result.top_reasons[0]}"
        raise DefenseBlockedError(msg, result=result)
    if result.action == "sanitize" and result.sanitized_output is not None:
        return result.sanitized_output
    return reply


async def scan_messages(
    messages: list[dict[str, Any]],
    *,
    session_id: str | None = None,
    user_id: str | None = None,
    settings: ForgeSettings | None = None,
) -> tuple[list[dict[str, Any]], DefenseResult | None]:
    """Pre-inference scan: optionally sanitize or block user input."""
    cfg = settings or get_settings()
    user_input = extract_user_input(messages)
    if not user_input.strip():
        return messages, None

    result = await analyze(
        user_input,
        session_id=session_id,
        metadata={"phase": "input", "user_id": user_id},
        settings=cfg,
    )
    audit_event(
        "autodefense_input",
        user_id=user_id,
        risk_score=result.risk_score,
        action=result.action,
        threat_types=result.threat_types,
        unavailable=result.unavailable,
    )

    sanitized = enforce_input(result)
    if sanitized is not None:
        return apply_input_sanitization(messages, sanitized), result
    return messages, result


async def scan_output(
    messages: list[dict[str, Any]],
    reply: str,
    *,
    session_id: str | None = None,
    user_id: str | None = None,
    settings: ForgeSettings | None = None,
) -> tuple[str, DefenseResult]:
    """Post-inference scan: optionally sanitize or block model output."""
    cfg = settings or get_settings()
    user_input = extract_user_input(messages)

    result = await analyze(
        user_input,
        model_output=reply,
        session_id=session_id,
        metadata={"phase": "output", "user_id": user_id},
        settings=cfg,
    )
    audit_event(
        "autodefense_output",
        user_id=user_id,
        risk_score=result.risk_score,
        action=result.action,
        threat_types=result.threat_types,
        unavailable=result.unavailable,
    )

    return enforce_output(result, reply), result
