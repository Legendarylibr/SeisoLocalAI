"""Metered inference helpers for the marketplace /v1 proxy."""

from __future__ import annotations

from typing import Any

from seiso.pay.ark import settle_split
from seiso.pay.pricing import quote_inference_tokens
from seiso.pay.store import debit_session


def estimate_tokens_from_messages(messages: list[Any]) -> int:
    """Rough prompt token estimate (word-ish); billing may use response usage."""
    total = 0
    for msg in messages or []:
        if isinstance(msg, dict):
            content = msg.get("content") or ""
        else:
            content = str(msg)
        total += max(1, len(str(content).split()))
    return total


def debit_inference(
    session_id: str,
    *,
    prompt_tokens: int,
    completion_tokens: int,
    flat_call: bool = False,
    data_dir=None,
) -> dict[str, Any]:
    quote = quote_inference_tokens(
        prompt_tokens,
        completion_tokens,
        flat_call=flat_call,
    )
    debit_session(
        session_id,
        compute_sats=int(quote["compute_sats"]),
        protocol_fee_sats=int(quote["protocol_fee_sats"]),
        data_dir=data_dir,
        reason="inference",
        meta=quote,
    )
    settlement = settle_split(
        compute_sats=int(quote["compute_sats"]),
        protocol_fee_sats=int(quote["protocol_fee_sats"]),
        session_id=session_id,
    )
    return {"quote": quote, "settlement": settlement.as_dict()}
