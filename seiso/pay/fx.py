"""Foreign-exchange quoting: sats → BTC → USD → USDC / ETH / stablecoin."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

_DEFAULT_BTC_USD_8 = 60_000_00_000_000  # 1 BTC = $60,000 (8 decimals)
_DEFAULT_ETH_USD_8 = 3_500_00_000_000   # 1 ETH  = $3,500  (8 decimals)
_DEFAULT_USDC_ATOMIC_PER_SAT = 5        # dev placeholder: 1 sat → 5 USDC atomic


@dataclass(frozen=True, slots=True)
class FxQuote:
    """Per-request FX quote: what this request costs in each denomination."""

    total_sats: int
    compute_sats: int
    protocol_fee_sats: int
    protocol_fee_bps: int
    btc_usd_8: int
    eth_usd_8: int
    usd_cents: int
    wei: int
    usdc_atomic: int
    eth_atomic: int  # wei

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def quote_fx(
    compute_sats: int,
    *,
    btc_usd_8: int | None = None,
    eth_usd_8: int | None = None,
    bps: int | None = None,
) -> FxQuote:
    """Convert sats compute to USD, wei, and USDC atomic (6 decimals)."""
    from seiso.pay.flags import protocol_fee_bps

    if compute_sats <= 0:
        compute_sats = 1
    bps = bps if bps is not None else protocol_fee_bps()
    fee = (compute_sats * int(bps)) // 10_000
    total = compute_sats + fee

    btc = btc_usd_8 or _DEFAULT_BTC_USD_8
    eth = eth_usd_8 or _DEFAULT_ETH_USD_8
    if btc <= 0:
        btc = _DEFAULT_BTC_USD_8
    if eth <= 0:
        eth = _DEFAULT_ETH_USD_8

    # 1 BTC = 100_000_000 sats
    # btc_usd_8: BTC price in USD with 8 decimals
    sats_per_btc = 100_000_000
    usd_cents = (total * btc) // (sats_per_btc * 1_000_000)  # cents

    # USDC atomic (6 decimals): 1 sat → atomic mapping
    import os

    raw = (os.environ.get("SEISO_PAY_X402_ATOMIC_PER_SAT") or str(_DEFAULT_USDC_ATOMIC_PER_SAT)).strip()
    try:
        atomic_per_sat = int(raw)
    except ValueError:
        atomic_per_sat = _DEFAULT_USDC_ATOMIC_PER_SAT
    if atomic_per_sat <= 0:
        atomic_per_sat = _DEFAULT_USDC_ATOMIC_PER_SAT
    usdc_atomic = total * atomic_per_sat

    # wei: 1 ETH = 10^18 wei; price from eth_usd_8
    if eth > 0 and usd_cents > 0:
        wei = (total * btc * 10**18) // (sats_per_btc * eth)
    else:
        wei = total * 100  # fallback
    if wei <= 0:
        wei = total * 100

    return FxQuote(
        total_sats=total,
        compute_sats=compute_sats,
        protocol_fee_sats=fee,
        protocol_fee_bps=bps,
        btc_usd_8=btc,
        eth_usd_8=eth,
        usd_cents=usd_cents,
        wei=wei,
        usdc_atomic=usdc_atomic,
        eth_atomic=wei,
    )
