"""Ark settlement interface — faucet now; Bark/Second wire when configured.

Also composes shared funding discovery (Ark + L402 + faucet). Live Ark and
live Lightning L402 are **not functional yet** — do not use for real funds.
``SEISO_PAY_L402_SIM=1`` (or faucet) enables simulated L402 fund/exchange.
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Literal

from seiso.pay.flags import (
    faucet_enabled,
    operator_ark,
    pay_settle_ready,
    payment_methods,
    protocol_treasury_ark,
)
from seiso.pay.l402 import funding_l402_block, l402_sim_enabled

SettleMode = Literal["faucet", "ark", "simulated", "l402"]


@dataclass(frozen=True, slots=True)
class SettlementReceipt:
    mode: SettleMode
    compute_sats: int
    protocol_fee_sats: int
    total_sats: int
    operator_destination: str
    protocol_destination: str
    status: str
    ts: float
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def funding_instructions(session_id: str, amount_sats: int) -> dict[str, Any]:
    """Payment instructions for buyers (Ark, L402, and/or faucet)."""
    ark_addr = operator_ark() or f"ark:pending:{session_id[:12]}"
    sim = l402_sim_enabled()
    out: dict[str, Any] = {
        "session_id": session_id,
        "amount_sats": amount_sats,
        "payment_methods": payment_methods(),
        "ark_address": ark_addr,
        "ln_invoice": None,
        "l402": funding_l402_block(session_id, amount_sats),
        "faucet_available": faucet_enabled(),
        "network": (os.environ.get("SEISO_ARK_NETWORK") or "signet").strip(),
        "status": "pending",
        "do_not_use_live_rails": True,
        "detail": (
            "Live Ark and live Lightning L402 are not functional yet — do not "
            "use for real funds. "
            + (
                "Simulated L402 fund/exchange is available (POST /pay/v1/sessions/fund/l402)."
                if sim
                else "Faucet/sim only for local smoke tests."
            )
        ),
    }
    if faucet_enabled():
        out["faucet_hint"] = "Dev faucet: seiso pay session fund --session ID --sats N --faucet"
    if sim:
        out["l402_hint"] = "Sim L402: seiso pay session fund --session ID --sats N --l402"
    return out


def settle_split(
    *,
    compute_sats: int,
    protocol_fee_sats: int,
    job_id: str | None = None,
    session_id: str | None = None,
) -> SettlementReceipt:
    """Release operator + protocol shares.

    Production: requires treasury + uses Ark client when ``SEISO_ARK_BACKEND`` set.
    Faucet/sim: records ledger-shaped receipt without chain IO.
    """
    ready, reason = pay_settle_ready()
    backend = (os.environ.get("SEISO_ARK_BACKEND") or "").strip().lower()
    total = int(compute_sats) + int(protocol_fee_sats)
    op_dest = operator_ark() or "operator:unset"
    proto_dest = protocol_treasury_ark() or "protocol:unset"

    if backend in {"bark", "second", "ark"}:
        if not ready:
            raise RuntimeError(reason)
        # Placeholder for Bark/Second SDK integration — fail clearly if not wired.
        raise RuntimeError(
            f"SEISO_ARK_BACKEND={backend} selected but Bark/Second client is not "
            "bundled yet. Use SEISO_PAY_FAUCET=1 for simulated settlement, or unset "
            "SEISO_ARK_BACKEND until the Ark wire is installed."
        )

    # Simulated / faucet settlement (phases 2–5 + tests)
    if not protocol_treasury_ark() and not faucet_enabled():
        raise RuntimeError(reason)

    mode: SettleMode = "faucet" if faucet_enabled() else "simulated"
    detail = (
        f"simulated split job={job_id or '-'} session={session_id or '-'} "
        f"operator={op_dest} protocol={proto_dest}"
    )
    return SettlementReceipt(
        mode=mode,
        compute_sats=int(compute_sats),
        protocol_fee_sats=int(protocol_fee_sats),
        total_sats=total,
        operator_destination=op_dest,
        protocol_destination=proto_dest or "faucet:treasury-placeholder",
        status="settled",
        ts=time.time(),
        detail=detail,
    )
