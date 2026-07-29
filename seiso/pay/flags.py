"""Opt-in marketplace flags — off by default; never gates local Seiso."""

from __future__ import annotations

import os

DEFAULT_PROTOCOL_FEE_BPS = 500  # 5%
MAX_PROTOCOL_FEE_BPS = 1000  # 10% hard clamp unless override


def _truthy(raw: str | None) -> bool:
    return (raw or "").strip().lower() in {"1", "true", "yes", "on"}


def pay_allowed() -> bool:
    """True only when operator explicitly enables the pay sidecar."""
    return _truthy(os.environ.get("SEISO_ALLOW_PAY"))


def require_pay_allowed() -> None:
    if not pay_allowed():
        raise RuntimeError(
            "Marketplace pay is disabled. Set SEISO_ALLOW_PAY=1 to opt in. "
            "Self-hosted Forge/CLI remain free and do not need this flag."
        )


def protocol_fee_bps() -> int:
    """Protocol fee in basis points (default 500 = 5%), clamped to 0–1000."""
    raw = (os.environ.get("SEISO_PROTOCOL_FEE_BPS") or "").strip()
    if not raw:
        return DEFAULT_PROTOCOL_FEE_BPS
    try:
        bps = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"SEISO_PROTOCOL_FEE_BPS must be an integer, got {raw!r}"
        ) from exc
    if bps < 0:
        raise ValueError("SEISO_PROTOCOL_FEE_BPS must be >= 0")
    override = _truthy(os.environ.get("SEISO_PROTOCOL_FEE_OVERRIDE"))
    if bps > MAX_PROTOCOL_FEE_BPS and not override:
        raise ValueError(
            f"SEISO_PROTOCOL_FEE_BPS={bps} exceeds max {MAX_PROTOCOL_FEE_BPS}; "
            "set SEISO_PROTOCOL_FEE_OVERRIDE=1 to allow (documented ops only)"
        )
    return bps


def protocol_treasury_ark() -> str:
    return (os.environ.get("SEISO_PROTOCOL_TREASURY_ARK") or "").strip()


def operator_ark() -> str:
    return (os.environ.get("SEISO_OPERATOR_ARK") or "").strip()


def forge_base_url() -> str:
    return (
        os.environ.get("SEISO_FORGE_URL") or "http://127.0.0.1:8765"
    ).rstrip("/")


def pay_settle_ready() -> tuple[bool, str]:
    """Whether paid settles may proceed (treasury required when pay enabled)."""
    if not pay_allowed():
        return False, "SEISO_ALLOW_PAY is not set"
    if not protocol_treasury_ark():
        return (
            False,
            "SEISO_PROTOCOL_TREASURY_ARK unset — refuse paid settles (fail closed)",
        )
    return True, "ok"


def faucet_enabled() -> bool:
    """Dev faucet for phases before real Ark (explicit opt-in)."""
    return _truthy(os.environ.get("SEISO_PAY_FAUCET"))
