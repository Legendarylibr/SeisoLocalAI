"""L402 (Lightning HTTP 402) payment rail — scaffolding only; not wired yet.

L402 joins HTTP 402 Payment Required with a Lightning invoice + macaroon
challenge so clients (including agents) can pay sats and retry with
``Authorization: L402 <macaroon>:<preimage>``.

Reference: https://lightningfaucet.com/learn/l402-payments-explained/

Status: **not functional yet — do not use.** Challenge minting, Lightning
invoice issuance, and preimage verification are not bundled. Prefer faucet/sim
for local smoke tests until an L402 server/client wire lands.
"""

from __future__ import annotations

import os
from typing import Any

from seiso.pay.flags import faucet_enabled

NOT_FUNCTIONAL_MSG = (
    "L402 settlement is not functional yet — do not use. "
    "Challenge minting / Lightning invoice issuance / preimage verification "
    "are not bundled. Use SEISO_PAY_FAUCET=1 for simulated funding, or leave "
    "L402 unset until the wire is installed. "
    "See https://lightningfaucet.com/learn/l402-payments-explained/"
)


def l402_advertised() -> bool:
    """True when operators opt into advertising L402 (default: on with pay).

    Set ``SEISO_PAY_L402=0`` to hide the rail from discovery/funding hints.
    Live L402 is still not wired either way.
    """
    raw = (os.environ.get("SEISO_PAY_L402") or "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def challenge_placeholder(
    *,
    session_id: str,
    amount_sats: int,
) -> dict[str, Any]:
    """Describe the intended L402 challenge shape without minting credentials."""
    return {
        "method": "l402",
        "status": "not_functional",
        "do_not_use": True,
        "session_id": session_id,
        "amount_sats": int(amount_sats),
        "http_status": 402,
        "www_authenticate": None,
        "macaroon": None,
        "invoice": None,
        "hint": (
            "When wired: client pays BOLT-11 invoice, retries with "
            "Authorization: L402 <macaroon>:<preimage>"
        ),
        "reference": "https://lightningfaucet.com/learn/l402-payments-explained/",
        "detail": NOT_FUNCTIONAL_MSG,
    }


def require_l402_ready() -> None:
    """Fail closed if code paths attempt live L402 settlement."""
    raise RuntimeError(NOT_FUNCTIONAL_MSG)


def funding_l402_block(session_id: str, amount_sats: int) -> dict[str, Any] | None:
    """L402 slice of session funding instructions, or None when not advertised."""
    if not l402_advertised():
        return None
    block = challenge_placeholder(session_id=session_id, amount_sats=amount_sats)
    if faucet_enabled():
        block["dev_note"] = (
            "Dev faucet is enabled for smoke tests; L402 itself remains unwired."
        )
    return block
