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
            "Marketplace pay is disabled (experimental). "
            "Set SEISO_ALLOW_PAY=1 to opt in. "
            "Not functional for real funds yet — faucet/sim only. "
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
        raise ValueError(f"SEISO_PROTOCOL_FEE_BPS must be an integer, got {raw!r}") from exc
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
    return (os.environ.get("SEISO_FORGE_URL") or "http://127.0.0.1:8765").rstrip("/")


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


def payment_methods() -> list[dict[str, str]]:
    """Advertised marketplace payment rails (discovery / docs).

    Live Ark and live Lightning L402 are **not functional yet** — do not use
    for real funds. Faucet and ``SEISO_PAY_L402_SIM`` credit sessions for smoke
    tests only.
    """
    from seiso.pay.l402 import l402_sim_enabled
    from seiso.pay.x402 import x402_advertised, x402_network, x402_sim_enabled

    l402_status = "sim" if l402_sim_enabled() else "not_functional"
    l402_detail = (
        "Simulated L402 fund/exchange (SEISO_PAY_L402_SIM or faucet); "
        "live Lightning not wired — "
        "https://lightningfaucet.com/learn/l402-payments-explained/"
        if l402_sim_enabled()
        else (
            "HTTP 402 + Lightning invoice + macaroon; live LN not wired — "
            "set SEISO_PAY_L402_SIM=1 for smoke tests; "
            "https://lightningfaucet.com/learn/l402-payments-explained/"
        )
    )
    x402_status = "sim" if x402_sim_enabled() else "not_functional"
    x402_detail = (
        f"Simulated x402 EVM fund/exchange on {x402_network()} "
        "(SEISO_PAY_X402_SIM or faucet); live USDC/facilitator not wired — "
        "https://x402.org/"
        if x402_sim_enabled()
        else (
            f"HTTP 402 + PAYMENT-REQUIRED/PAYMENT-SIGNATURE on {x402_network()}; "
            "live EVM not wired — set SEISO_PAY_X402_SIM=1 for smoke tests; "
            "https://x402.org/"
        )
    )
    methods: list[dict[str, str]] = [
        {
            "id": "ark",
            "label": "Ark pay-in",
            "status": "not_functional",
            "detail": "Operator/treasury Ark addresses; Bark/Second client not bundled",
        },
        {
            "id": "l402",
            "label": "L402 (Lightning HTTP 402)",
            "status": l402_status,
            "detail": l402_detail,
        },
        {
            "id": "x402",
            "label": "x402 (EVM HTTP 402 / USDC)",
            "status": x402_status,
            "detail": x402_detail,
        },
    ]
    # Allow operators to hide rails from discovery.
    l402_raw = (os.environ.get("SEISO_PAY_L402") or "1").strip().lower()
    if l402_raw in {"0", "false", "no", "off"}:
        methods = [m for m in methods if m["id"] != "l402"]
    if not x402_advertised():
        methods = [m for m in methods if m["id"] != "x402"]
    if faucet_enabled():
        methods.append(
            {
                "id": "faucet",
                "label": "Dev faucet",
                "status": "dev_only",
                "detail": "SEISO_PAY_FAUCET=1 — never enable on a public market",
            }
        )
    return methods
