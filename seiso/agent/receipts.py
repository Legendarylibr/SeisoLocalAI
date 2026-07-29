"""Generic agent receipts with Buzz-compatible shape.

Receipts are safe to post to a channel (no secrets). Buzz agents can post them
via buzz-cli; other harnesses can log or store the same JSON.

Seiso never posts to Buzz itself — the agent harness does. These helpers
scrub secrets so a mistaken ``**fields`` cannot leak tokens/nsecs.
"""

from __future__ import annotations

from typing import Any

# Exact keys and substrings refused in channel-facing receipts.
_FORBIDDEN_KEYS = frozenset(
    {
        "token",
        "nsec",
        "private_key",
        "buzz_private_key",
        "mesh_token",
        "seiso_mesh_token",
        "token_fingerprint",
        "authorization",
        "password",
        "secret",
        "api_key",
        "access_token",
        "refresh_token",
    }
)
_FORBIDDEN_SUBSTR = (
    "token",
    "nsec",
    "secret",
    "password",
    "private_key",
    "authorization",
)


def _is_forbidden_field(key: str) -> bool:
    lowered = key.strip().lower()
    if lowered in _FORBIDDEN_KEYS:
        return True
    return any(part in lowered for part in _FORBIDDEN_SUBSTR)


def agent_receipt(
    *,
    role: str,
    status: str,
    surface: str = "agent",
    **fields: Any,
) -> dict[str, Any]:
    """Build a harness-agnostic status receipt (channel-safe)."""
    out: dict[str, Any] = {
        "role": role,
        "status": status,
        "surface": surface,
        "buzz_compatible": True,
    }
    for key, value in fields.items():
        if value is None:
            continue
        if _is_forbidden_field(str(key)):
            continue
        out[key] = value
    return out


def buzz_compatible_receipt(receipt: dict[str, Any]) -> dict[str, Any]:
    """Alias helper — same payload under the historical ``buzz_receipt`` key shape.

    Callers that already post ``buzz_receipt`` to a Buzz channel keep working;
    generic agents can use the same dict without a Buzz runtime.
    """
    return {k: v for k, v in receipt.items() if not _is_forbidden_field(str(k))}


def channel_safe_plan_view(plan: dict[str, Any]) -> dict[str, Any]:
    """Strip secret-binding material from a plan before printing / pasting."""
    return {
        k: v
        for k, v in plan.items()
        if k != "token_fingerprint" and not _is_forbidden_field(str(k))
    }
