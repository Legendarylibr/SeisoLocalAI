"""Ark / BTC-L2 per-request voucher (descriptor template).

Ark vtxos are covenant-bound UTXOs. Seiso does not bundle a Bark client;
this module emits a **miniscript descriptor** and a voucher JSON that an
operator's Ark wallet can materialise. Settlement remains fail-closed
until ``SEISO_ARK_BACKEND`` is wired.

Policy (2-path, same as the HTLC):
    or(
      and(pk(operator), sha256(payment_hash)),
      and(pk(buyer), after(locktime))
    )
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


def miniscript(
    *,
    operator_xonly_or_desc: str,
    buyer_xonly_or_desc: str,
    payment_hash_hex: str,
    locktime: int,
) -> str:
    if len(payment_hash_hex) != 64:
        raise ValueError("payment_hash_hex must be 32-byte hex")
    if locktime < 0:
        raise ValueError("locktime must be >= 0")
    op = operator_xonly_or_desc.strip()
    by = buyer_xonly_or_desc.strip()
    return (
        f"wsh(or_d("
        f"and_v(v:pk({op}),sha256({payment_hash_hex})),"
        f"and_v(v:pk({by}),after({int(locktime)}))"
        f"))"
    )


@dataclass(frozen=True, slots=True)
class ArkVoucher:
    amount_sats: int
    payment_hash_hex: str
    locktime: int
    descriptor: str
    network: str
    note: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_voucher(
    *,
    amount_sats: int,
    payment_hash_hex: str,
    operator_key: str,
    buyer_key: str,
    locktime: int,
    network: str = "signet",
) -> ArkVoucher:
    if amount_sats <= 0:
        raise ValueError("amount_sats must be > 0")
    desc = miniscript(
        operator_xonly_or_desc=operator_key,
        buyer_xonly_or_desc=buyer_key,
        payment_hash_hex=payment_hash_hex,
        locktime=locktime,
    )
    return ArkVoucher(
        amount_sats=int(amount_sats),
        payment_hash_hex=payment_hash_hex,
        locktime=int(locktime),
        descriptor=desc,
        network=network,
        note=(
            "Descriptor only — not a live Ark vtxo. Wire SEISO_ARK_BACKEND "
            "and a Bark/Second client before using real funds."
        ),
    )
