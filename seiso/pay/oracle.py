"""Market prices for ETH/USD and BTC/USD (8 decimals).

Trusted sources, in order:
1. Explicit constructor / function args (tests)
2. Env ``SEISO_PAY_ETH_USD_8`` / ``SEISO_PAY_BTC_USD_8``
3. Optional HTTP JSON ``SEISO_PAY_ORACLE_URL`` (must return
   ``{"eth_usd_8": int, "btc_usd_8": int, "updated_at": unix}``)
4. Fail closed — never invent a price

On-chain the same numbers live in ``SeisoPriceOracle`` (Chainlink feed
preferred, ``setFallbackPrice`` for updates).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

DEFAULT_MAX_STALENESS_S = 3600


@dataclass(frozen=True, slots=True)
class OraclePrices:
    eth_usd_8: int
    btc_usd_8: int
    updated_at: float
    source: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "eth_usd_8": self.eth_usd_8,
            "btc_usd_8": self.btc_usd_8,
            "eth_usd": self.eth_usd_8 / 1e8,
            "btc_usd": self.btc_usd_8 / 1e8,
            "updated_at": self.updated_at,
            "source": self.source,
        }


def _truthy(raw: str | None) -> bool:
    return (raw or "").strip().lower() in {"1", "true", "yes", "on"}


def max_staleness_s() -> int:
    raw = (os.environ.get("SEISO_PAY_ORACLE_MAX_STALE_S") or "").strip()
    if not raw:
        return DEFAULT_MAX_STALENESS_S
    return max(30, int(raw))


def _from_env() -> OraclePrices | None:
    eth = (os.environ.get("SEISO_PAY_ETH_USD_8") or "").strip()
    btc = (os.environ.get("SEISO_PAY_BTC_USD_8") or "").strip()
    if not eth or not btc:
        return None
    ts_raw = (os.environ.get("SEISO_PAY_ORACLE_UPDATED_AT") or "").strip()
    updated = float(ts_raw) if ts_raw else time.time()
    return OraclePrices(
        eth_usd_8=int(eth),
        btc_usd_8=int(btc),
        updated_at=updated,
        source="env",
    )


def _oracle_url_allowed(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return False
    host = (parsed.hostname or "").lower()
    # Local-first: loopback or explicitly allowlisted hosts.
    if host in {"127.0.0.1", "localhost", "::1"}:
        return True
    allow = (os.environ.get("SEISO_PAY_ORACLE_ALLOW_HOSTS") or "").strip()
    if not allow:
        return False
    allowed = {h.strip().lower() for h in allow.split(",") if h.strip()}
    return host in allowed


def _from_http() -> OraclePrices | None:
    url = (os.environ.get("SEISO_PAY_ORACLE_URL") or "").strip()
    if not url:
        return None
    if not _oracle_url_allowed(url):
        raise RuntimeError(
            "SEISO_PAY_ORACLE_URL host not allowlisted (loopback or SEISO_PAY_ORACLE_ALLOW_HOSTS)"
        )
    import httpx

    resp = httpx.get(url, timeout=5.0)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, dict):
        raise RuntimeError("oracle JSON must be an object")
    return OraclePrices(
        eth_usd_8=int(data["eth_usd_8"]),
        btc_usd_8=int(data["btc_usd_8"]),
        updated_at=float(data.get("updated_at") or time.time()),
        source=str(data.get("source") or url),
    )


def load_prices(*, allow_stale: bool = False) -> OraclePrices:
    prices = _from_env()
    if prices is None and (os.environ.get("SEISO_PAY_ORACLE_URL") or "").strip():
        prices = _from_http()
    if prices is None:
        raise RuntimeError(
            "No ETH/BTC USD price configured. Set SEISO_PAY_ETH_USD_8 and "
            "SEISO_PAY_BTC_USD_8 (8 decimals, Chainlink-style) or "
            "SEISO_PAY_ORACLE_URL. On-chain, call SeisoPriceOracle."
            "setFallbackPrice or bind a Chainlink feed."
        )
    if prices.eth_usd_8 <= 0 or prices.btc_usd_8 <= 0:
        raise RuntimeError("oracle prices must be > 0")
    age = time.time() - prices.updated_at
    if not allow_stale and age > max_staleness_s():
        raise RuntimeError(f"oracle price stale ({int(age)}s > {max_staleness_s()}s)")
    return prices


def load_prices_or_none() -> OraclePrices | None:
    try:
        return load_prices()
    except Exception:
        return None
