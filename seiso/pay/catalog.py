"""Operator catalog listings for distributed inference and training.

Settlement rails stay fail-closed. There is no Seiso token.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from seiso.agent.tasks import parse_task_kind
from seiso.pay.flags import faucet_enabled, pay_allowed, payment_methods, protocol_treasury_ark
from seiso.pay.pricing import JOB_TYPES, fee_split, price_for_job, quote_inference_tokens
from seiso.routing.external import is_loopback_url

LISTING_KINDS = frozenset(JOB_TYPES) | frozenset({"inference"})

# Keys that must never appear on a listing or quote (no platform coin).
FORBIDDEN_LISTING_KEYS = frozenset(
    {
        "seiso_token",
        "SEISO_TOKEN",
        "token_ticker",
        "coin",
        "airdrop",
    }
)


@dataclass(frozen=True, slots=True)
class Listing:
    """One operator offering. ``loopback`` listings are always free."""

    kind: str
    label: str
    operator_id: str
    compute_sats: int
    gpu_class: str = ""
    model_or_preset: str = ""
    loopback: bool = False
    endpoint: str = ""

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in FORBIDDEN_LISTING_KEYS:
            data.pop(key, None)
        return data


def parse_listing_kind(raw: str) -> str:
    kind = parse_task_kind(raw)
    if kind.value in {"chat", "code", "embed", "draft", "target"}:
        return "inference"
    if kind.value not in LISTING_KINDS:
        raise ValueError(f"unknown listing kind {raw!r}")
    return kind.value


def live_settle_allowed(
    *,
    ark_live: bool = False,
    l402_live: bool = False,
    x402_live: bool = False,
    treasury_set: bool | None = None,
    faucet: bool | None = None,
) -> bool:
    """Live rails are not wired. Always False unless a future caller opts into both."""
    if faucet is None:
        faucet = faucet_enabled()
    if treasury_set is None:
        treasury_set = bool(protocol_treasury_ark())
    if faucet:
        return False
    if not treasury_set:
        return False
    return bool(ark_live or l402_live or x402_live)


def quote_listing(
    listing: Listing,
    *,
    bps: int | None = None,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> dict[str, Any]:
    """Quote a listing. Loopback / localhost endpoints are 0 sats."""
    kind = parse_listing_kind(listing.kind)
    loopback = listing.loopback or is_loopback_url(listing.endpoint)
    if loopback:
        split = fee_split(0, bps=0)
        out = split.as_dict()
        out.update(
            {
                "job_type": kind,
                "listing": listing.as_dict(),
                "rails": payment_methods(),
                "live_settle_allowed": False,
                "loopback": True,
                "price_sats": 0,
            }
        )
        for key in FORBIDDEN_LISTING_KEYS:
            out.pop(key, None)
        return out

    if kind == "inference" and (prompt_tokens or completion_tokens):
        out = quote_inference_tokens(prompt_tokens, completion_tokens, bps=bps)
    elif kind == "inference":
        split = fee_split(max(0, int(listing.compute_sats)), bps=bps)
        out = split.as_dict()
        out["job_type"] = "inference"
    else:
        compute = int(listing.compute_sats)
        if compute <= 0:
            compute = price_for_job(kind, preset=listing.model_or_preset or None)
        split = fee_split(compute, bps=bps)
        out = split.as_dict()
        out["job_type"] = kind
        out["preset"] = listing.model_or_preset or None

    out["listing"] = listing.as_dict()
    out["rails"] = payment_methods()
    out["live_settle_allowed"] = live_settle_allowed()
    out["loopback"] = False
    out["price_sats"] = int(out["total_sats"])
    out["pay_allowed"] = pay_allowed()
    for key in FORBIDDEN_LISTING_KEYS:
        out.pop(key, None)
        if isinstance(out.get("listing"), dict):
            out["listing"].pop(key, None)
    return out
