"""Oracle load + staleness."""

from __future__ import annotations

import time

import pytest

from seiso.pay.oracle import load_prices, load_prices_or_none


def test_env_prices(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEISO_PAY_ETH_USD_8", str(2500 * 10**8))
    monkeypatch.setenv("SEISO_PAY_BTC_USD_8", str(100_000 * 10**8))
    monkeypatch.setenv("SEISO_PAY_ORACLE_UPDATED_AT", str(time.time()))
    p = load_prices()
    assert p.eth_usd_8 == 2500 * 10**8
    assert p.btc_usd_8 == 100_000 * 10**8
    assert p.source == "env"


def test_stale_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEISO_PAY_ETH_USD_8", "1")
    monkeypatch.setenv("SEISO_PAY_BTC_USD_8", "1")
    monkeypatch.setenv("SEISO_PAY_ORACLE_UPDATED_AT", str(time.time() - 10_000))
    monkeypatch.setenv("SEISO_PAY_ORACLE_MAX_STALE_S", "60")
    with pytest.raises(RuntimeError, match="stale"):
        load_prices()


def test_missing_prices(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SEISO_PAY_ETH_USD_8", raising=False)
    monkeypatch.delenv("SEISO_PAY_BTC_USD_8", raising=False)
    monkeypatch.delenv("SEISO_PAY_ORACLE_URL", raising=False)
    assert load_prices_or_none() is None
    with pytest.raises(RuntimeError, match="No ETH/BTC"):
        load_prices()


def test_oracle_url_requires_allowlist(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SEISO_PAY_ETH_USD_8", raising=False)
    monkeypatch.delenv("SEISO_PAY_BTC_USD_8", raising=False)
    monkeypatch.setenv("SEISO_PAY_ORACLE_URL", "https://evil.example/oracle")
    monkeypatch.delenv("SEISO_PAY_ORACLE_ALLOW_HOSTS", raising=False)
    with pytest.raises(RuntimeError, match="allowlisted"):
        load_prices()
