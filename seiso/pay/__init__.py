"""Opt-in sats marketplace for remote inference / finetune / RL.

Self-hosted Seiso never imports this on the hot path. Enable with
``SEISO_ALLOW_PAY=1`` and run ``seiso pay serve`` (sidecar).
"""

from __future__ import annotations

from seiso.pay.catalog import Listing, live_settle_allowed, quote_listing
from seiso.pay.flags import (
    DEFAULT_PROTOCOL_FEE_BPS,
    MAX_PROTOCOL_FEE_BPS,
    pay_allowed,
    protocol_fee_bps,
    protocol_treasury_ark,
    require_pay_allowed,
)
from seiso.pay.pricing import FeeSplit, quote_compute
from seiso.pay.x402 import (
    USDC_BY_NETWORK,
    list_supported_networks,
    x402_advertised,
    x402_asset,
    x402_network,
    x402_sim_enabled,
)

__all__ = [
    "DEFAULT_PROTOCOL_FEE_BPS",
    "MAX_PROTOCOL_FEE_BPS",
    "FeeSplit",
    "Listing",
    "USDC_BY_NETWORK",
    "list_supported_networks",
    "live_settle_allowed",
    "pay_allowed",
    "protocol_fee_bps",
    "protocol_treasury_ark",
    "quote_compute",
    "quote_listing",
    "require_pay_allowed",
    "x402_advertised",
    "x402_asset",
    "x402_network",
    "x402_sim_enabled",
]
