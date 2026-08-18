"""FX math for per-request ETH / USDC quotes.

Must stay bit-identical to ``contracts/src/SeisoPayRouter.sol``:

    usdc_6 = ceil(sats * btc_usd_8 / 10**10)
    wei    = ceil(sats * btc_usd_8 * 10**10 / eth_usd_8)

Prices are USD per 1 BTC or 1 ETH with **8 decimals** (Chainlink).
Sats are the marketplace unit (1 BTC = 100_000_000 sats).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from seiso.pay.pricing import fee_split

SATS_PER_BTC = 100_000_000
PRICE_DECIMALS = 8
USDC_DECIMALS = 6
WEI_PER_ETH = 10**18
PRICE_SCALE = 10**PRICE_DECIMALS  # 1e8


def ceil_div(num: int, den: int) -> int:
    if den <= 0:
        raise ValueError("denominator must be > 0")
    if num < 0:
        raise ValueError("numerator must be >= 0")
    return (num + den - 1) // den


def protocol_fee_sats(compute_sats: int, bps: int) -> int:
    if bps < 0 or bps > 1000:
        raise ValueError("fee bps out of range")
    return (int(compute_sats) * int(bps) + 9_999) // 10_000


def total_sats(compute_sats: int, bps: int) -> int:
    return int(compute_sats) + protocol_fee_sats(compute_sats, bps)


def required_usdc_atomic(sats: int, btc_usd_8: int) -> int:
    """USDC atomic (6 decimals) for ``sats`` at ``btc_usd_8``."""
    if sats < 0 or btc_usd_8 <= 0:
        raise ValueError("sats >= 0 and btc_usd_8 > 0 required")
    return ceil_div(int(sats) * int(btc_usd_8), 10**10)


def required_wei(sats: int, btc_usd_8: int, eth_usd_8: int) -> int:
    """wei for ``sats`` given BTC/USD and ETH/USD (both 8 decimals)."""
    if sats < 0 or btc_usd_8 <= 0 or eth_usd_8 <= 0:
        raise ValueError("sats >= 0 and positive oracle prices required")
    return ceil_div(int(sats) * int(btc_usd_8) * (10**10), int(eth_usd_8))


def split_atomic(total_atomic: int, compute_sats: int, fee_sats: int) -> tuple[int, int]:
    """Split paid atomic units in the same ratio as sats (treasury gets ceil)."""
    due = compute_sats + fee_sats
    if due <= 0:
        raise ValueError("due sats must be > 0")
    treasury = ceil_div(int(total_atomic) * int(fee_sats), due)
    operator = int(total_atomic) - treasury
    if operator < 0:
        raise ValueError("split underflow")
    return operator, treasury


@dataclass(frozen=True, slots=True)
class FxQuote:
    compute_sats: int
    protocol_fee_sats: int
    total_sats: int
    protocol_fee_bps: int
    btc_usd_8: int
    eth_usd_8: int
    usdc_atomic: int
    wei: int
    operator_usdc: int
    treasury_usdc: int
    operator_wei: int
    treasury_wei: int

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["eth_wei"] = self.wei
        data["eth"] = self.wei / WEI_PER_ETH
        data["usdc"] = self.usdc_atomic / (10**USDC_DECIMALS)
        data["btc_usd"] = self.btc_usd_8 / PRICE_SCALE
        data["eth_usd"] = self.eth_usd_8 / PRICE_SCALE
        return data


def quote_fx(
    compute_sats: int,
    *,
    btc_usd_8: int,
    eth_usd_8: int,
    bps: int | None = None,
) -> FxQuote:
    split = fee_split(int(compute_sats), bps=bps)
    usdc = required_usdc_atomic(split.total_sats, btc_usd_8)
    wei = required_wei(split.total_sats, btc_usd_8, eth_usd_8)
    op_u, tr_u = split_atomic(usdc, split.compute_sats, split.protocol_fee_sats)
    op_w, tr_w = split_atomic(wei, split.compute_sats, split.protocol_fee_sats)
    return FxQuote(
        compute_sats=split.compute_sats,
        protocol_fee_sats=split.protocol_fee_sats,
        total_sats=split.total_sats,
        protocol_fee_bps=split.protocol_fee_bps,
        btc_usd_8=int(btc_usd_8),
        eth_usd_8=int(eth_usd_8),
        usdc_atomic=usdc,
        wei=wei,
        operator_usdc=op_u,
        treasury_usdc=tr_u,
        operator_wei=op_w,
        treasury_wei=tr_w,
    )
