"""Generic agent receipts with Buzz-compatible shape.

Receipts are safe to post to a channel (no secrets). Buzz agents can post them
via buzz-cli; other harnesses can log or store the same JSON.
"""

from __future__ import annotations

from typing import Any


def agent_receipt(
    *,
    role: str,
    status: str,
    surface: str = "agent",
    **fields: Any,
) -> dict[str, Any]:
    """Build a harness-agnostic status receipt."""
    out: dict[str, Any] = {
        "role": role,
        "status": status,
        "surface": surface,
        "buzz_compatible": True,
    }
    for key, value in fields.items():
        if value is not None:
            out[key] = value
    return out


def buzz_compatible_receipt(receipt: dict[str, Any]) -> dict[str, Any]:
    """Alias helper — same payload under the historical ``buzz_receipt`` key shape.

    Callers that already post ``buzz_receipt`` to a Buzz channel keep working;
    generic agents can use the same dict without a Buzz runtime.
    """
    return dict(receipt)
