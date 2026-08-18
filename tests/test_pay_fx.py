"""FX math must match SeisoPayRouter.sol."""

from __future__ import annotations

import pytest

from seiso.pay.fx import (
    ceil_div,
    protocol_fee_sats,
    quote_fx,
    required_usdc_atomic,
    required_wei,
    split_atomic,
    total_sats,
)

BTC = 100_000 * 10**8
ETH = 2_500 * 10**8


def test_ceil_div() -> None:
    assert ceil_div(0, 3) == 0
    assert ceil_div(1, 3) == 1
    assert ceil_div(3, 3) == 1
    assert ceil_div(4, 3) == 2
    with pytest.raises(ValueError):
        ceil_div(1, 0)


def test_fee_matches_solidity_and_python_split() -> None:
    assert protocol_fee_sats(10, 500) == 1
    assert protocol_fee_sats(10_000, 500) == 500
    assert total_sats(10_000, 500) == 10_500


def test_usdc_and_wei_vectors() -> None:
    assert required_usdc_atomic(10_500, BTC) == 10_500_000
    assert required_wei(10_500, BTC, ETH) == 4_200_000_000_000_000


def test_quote_fx_split_sums() -> None:
    q = quote_fx(10_000, btc_usd_8=BTC, eth_usd_8=ETH, bps=500)
    assert q.total_sats == 10_500
    assert q.usdc_atomic == 10_500_000
    assert q.wei == 4_200_000_000_000_000
    assert q.operator_usdc + q.treasury_usdc == q.usdc_atomic
    assert q.operator_wei + q.treasury_wei == q.wei
    assert q.treasury_usdc == split_atomic(q.usdc_atomic, 10_000, 500)[1]


def test_eth_price_update_changes_wei() -> None:
    cheap = quote_fx(10_000, btc_usd_8=BTC, eth_usd_8=ETH, bps=500)
    dear = quote_fx(10_000, btc_usd_8=BTC, eth_usd_8=5_000 * 10**8, bps=500)
    assert dear.wei < cheap.wei
    assert dear.usdc_atomic == cheap.usdc_atomic


def test_btc_price_update_changes_both() -> None:
    low = quote_fx(10_000, btc_usd_8=50_000 * 10**8, eth_usd_8=ETH, bps=500)
    high = quote_fx(10_000, btc_usd_8=BTC, eth_usd_8=ETH, bps=500)
    assert high.usdc_atomic > low.usdc_atomic
    assert high.wei > low.wei


def test_rejects_bad_oracle() -> None:
    with pytest.raises(ValueError):
        required_wei(1, 0, ETH)
    with pytest.raises(ValueError):
        required_usdc_atomic(1, 0)
