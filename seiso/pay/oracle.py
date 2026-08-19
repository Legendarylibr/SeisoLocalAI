"""Price oracle for BTC/USD and ETH/USD rates.

Sim mode returns env-configured or hardcoded defaults. Live mode reads from a
SeisoPriceOracle contract on-chain (future).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_DEFAULT_BTC_USD_8 = 60_000_00_000_000
_DEFAULT_ETH_USD_8 = 3_500_00_000_000


@dataclass(frozen=True, slots=True)
class OraclePrices:
    """Current prices from the oracle."""

    btc_usd_8: int
    eth_usd_8: int
    timestamp: int
    source: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def as_response(self) -> dict[str, Any]:
        return {
            "btc_usd_8": self.btc_usd_8,
            "eth_usd_8": self.eth_usd_8,
            "timestamp": self.timestamp,
            "source": self.source,
        }


def default_prices() -> OraclePrices:
    """Fallback prices from environment or hardcoded defaults."""
    import time

    btc = int(os.environ.get("SEISO_ORACLE_BTC_USD_8") or str(_DEFAULT_BTC_USD_8))
    eth = int(os.environ.get("SEISO_ORACLE_ETH_USD_8") or str(_DEFAULT_ETH_USD_8))
    return OraclePrices(
        btc_usd_8=max(btc, 1),
        eth_usd_8=max(eth, 1),
        timestamp=int(time.time()),
        source="env_default",
    )


def load_prices(data_dir: Path | None = None) -> OraclePrices:
    """Load prices from oracle cache file or env defaults.

    Live oracle reads from chain via ``SeisoPriceOracle`` when wired.
    """
    cache = None
    if data_dir:
        cache_file = data_dir / "pay" / "oracle_cache.json"
        if cache_file.is_file():
            try:
                raw = json.loads(cache_file.read_text(encoding="utf-8"))
                ts = int(raw.get("timestamp") or 0)
                age = int(time.time()) - ts
                if age < 300:  # 5 min cache
                    return OraclePrices(
                        btc_usd_8=int(raw.get("btc_usd_8", _DEFAULT_BTC_USD_8)),
                        eth_usd_8=int(raw.get("eth_usd_8", _DEFAULT_ETH_USD_8)),
                        timestamp=ts,
                        source=raw.get("source", "cache"),
                    )
            except (json.JSONDecodeError, KeyError, ValueError):
                pass
    return default_prices()


import time  # noqa: E402 — used for caching, placed after guard-free deps


def cache_prices(prices: OraclePrices, data_dir: Path | None = None) -> None:
    """Write oracle prices to the local cache file."""
    if data_dir:
        pay_dir = data_dir / "pay"
        pay_dir.mkdir(parents=True, exist_ok=True)
        cache_file = pay_dir / "oracle_cache.json"
        cache_file.write_text(
            json.dumps(prices.as_response(), indent=2) + "\n",
            encoding="utf-8",
        )
