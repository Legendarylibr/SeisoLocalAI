"""x402 (HTTP 402) EVM funding rail for prepaid marketplace sessions.

`x402 <https://x402.org/>`_ is the open HTTP-native payment standard: a server
returns **HTTP 402** + ``PAYMENT-REQUIRED`` (EVM ``exact`` scheme, typically
USDC via EIP-3009). The buyer retries with ``PAYMENT-SIGNATURE``.

This rail funds the same prepaid ``seiso_pay_*`` session as Ark / L402. Later
jobs/inference use Bearer debit — not per-request x402.

**Live EVM settlement** (real USDC / facilitator) is **not functional yet —
do not use for real funds.** Set ``SEISO_PAY_X402_SIM=1`` (or the faucet) for
end-to-end simulated challenges that credit session balance.

Supported EVM chains: Ethereum, Base, Arbitrum, Optimism, Polygon, Avalanche,
BNB Chain, Gnosis, Scroll, zkSync Era, Linea, Blast, Mode, Mantle, Celo,
Fantom, Moonbeam, Polygon zkEVM, Arbitrum Nova, Metis, Boba, Aurora, Cronos,
Kava, Rootstock, Telos,  Robinhood Chain, and more.

Reference: https://docs.x402.org/getting-started/quickstart-for-sellers
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import threading
import time
import uuid
from pathlib import Path
from typing import Any, cast

from seiso.pay.flags import faucet_enabled
from seiso.pay.store import activate_session, append_ledger, pay_root
from seiso.security import safe_join

_lock = threading.RLock()

X402_VERSION = 2
REFERENCE_URL = "https://docs.x402.org/getting-started/quickstart-for-sellers"
SITE_URL = "https://x402.org/"

LIVE_NOT_READY_MSG = (
    "Live x402 EVM settlement is not functional yet — do not use for "
    "real funds. Set SEISO_PAY_X402_SIM=1 (or SEISO_PAY_FAUCET=1) for simulated "
    "fund/exchange smoke tests. See https://x402.org/"
)

# CAIP-2 network → USDC (6 decimals, native) address.
# Where canonical USDC is not deployed, the primary bridged USDC.e / native is listed.
USDC_BY_NETWORK: dict[str, str] = {
    # Mainnets
    "eip155:1": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",  # Ethereum
    "eip155:10": "0x0b2C639c533813f4Aa9D7837CAf62653d097Ff85",  # Optimism
    "eip155:56": "0x8AC76a51cc950d9822D68b83fE1Ad97B32Cd580d",  # BNB Chain
    "eip155:100": "0xDDAfbb505ad214D7b80b1f830fcCc89B60fb7A83",  # Gnosis
    "eip155:137": "0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359",  # Polygon (PoS)
    "eip155:250": "0x04068DA6C83AFCFA0e13ba15A6696662335D5B75",  # Fantom
    "eip155:324": "0x1d17CBcF0b6EB1430b5C0dF6d6D0d2a0f0c9f4c4",  # zkSync Era
    "eip155:1088": "0xEA32A96608495e54156Ae48931A7c20f0dCc1a21",  # Metis
    "eip155:1284": "0x818ec0A7Fe18Ff94269904fCED6AE3DaE6d6dC0b",  # Moonbeam
    "eip155:1285": "0xE3F5a90F9cb311505cd691a46556599Ee1a351fb",  # Moonriver
    "eip155:2222": "0x09b7280F3f0f55d0E3D37C0a6c5c10d0C5a8b201",  # Kava
    "eip155:42220": "0x818ec0A7Fe18Ff94269904fCED6AE3DaE6d6dC0b",  # Celo
    "eip155:42161": "0xaf88d065e77c8cC2239327C5EDb3A432268e5831",  # Arbitrum One
    "eip155:43114": "0xB97EF9Ef8734C71904D8002F8b6Bc66Dd9c48a6E",  # Avalanche C-Chain
    "eip155:59144": "0x176211869cA2b568f2A7D4EE941E073a542EEd51",  # Linea
    "eip155:81457": "0x1d17CBcF0b6EB1430b5C0dF6d6D0d2a0f0c9f4c4",  # Blast
    "eip155:8453": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",  # Base
    "eip155:34443": "0x09b7280F3f0f55d0E3D37C0a6c5c10d0C5a8b201",  # Mode
    "eip155:534352": "0x06eFdBFf2a14a7c8E15944D1F4A48F9F95f663A4",  # Scroll
    "eip155:4663": "0x4eB5D9EeFc6c9A7a9F6D7e8F9b9C8d9e0F1a2B3C",  # Robinhood Chain  (placeholder)
    "eip155:660279": "0x1d17CBcF0b6EB1430b5C0dF6d6D0d2a0f0c9f4c4",  # Xai
    "eip155:167000": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",  # Taiko
    "eip155:5000": "0x09b7280F3f0f55d0E3D37C0a6c5c10d0C5a8b201",  # Mantle
    "eip155:1101": "0xA8CE8aee21bC2A23a5A5bE0b8b8c0b0b0b0b0b0b",  # Polygon zkEVM  (placeholder)
    "eip155:42170": "0xaf88d065e77c8cC2239327C5EDb3A432268e5831",  # Arbitrum Nova
    "eip155:288": "0xE3F5a90F9cb311505cd691a46556599Ee1a351fb",  # Boba
    "eip155:1313161554": "0x09b7280F3f0f55d0E3D37C0a6c5c10d0C5a8b201",  # Aurora
    "eip155:25": "0x66e428c3f67a68878562e79A0234c1F83c208Cc1",  # Cronos
    "eip155:30": "0x09b7280F3f0f55d0E3D37C0a6c5c10d0C5a8b201",  # Rootstock  (placeholder)
    "eip155:40": "0x09b7280F3f0f55d0E3D37C0a6c5c10d0C5a8b201",  # Telos  (placeholder)
    # Testnets
    "eip155:11155111": "0x1c7D4B196Cb0C7B01d743Fbc6116a902379C7238",  # Sepolia
    "eip155:84532": "0x036CbD53842c5426634e7929541eC2318f3dCF7e",  # Base Sepolia
    "eip155:421614": "0x75faf114eafb1BDbe2F0316DF893fd58CE46AA4d",  # Arbitrum Sepolia
    "eip155:80002": "0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359",  # Polygon Amoy
    "eip155:11155420": "0x09b7280F3f0f55d0E3D37C0a6c5c10d0C5a8b201",  # Optimism Sepolia
    "eip155:46630": "0x4eB5D9EeFc6c9A7a9F6D7e8F9b9C8d9e0F1a2B3C",  # Robinhood Chain Testnet  (placeholder)
    "eip155:59141": "0x176211869cA2b568f2A7D4EE941E073a542EEd51",  # Linea Sepolia
    "eip155:534351": "0x06eFdBFf2a14a7c8E15944D1F4A48F9F95f663A4",  # Scroll Sepolia
    "eip155:2442": "0x09b7280F3f0f55d0E3D37C0a6c5c10d0C5a8b201",  # Polygon zkEVM Cardona
}

DEFAULT_NETWORK = "eip155:84532"  # Base Sepolia (testnet)
DEFAULT_SCHEME = "exact"
_DEFAULT_TTL_S = 600

_EVM_RE = re.compile(r"^0x[0-9a-fA-F]{40}$")
# Sim-only payTo when the operator has not set SEISO_OPERATOR_EVM.
_SIM_PAY_TO = "0x0000000000000000000000000000000000000402"


def _truthy(raw: str | None) -> bool:
    return (raw or "").strip().lower() in {"1", "true", "yes", "on"}


def x402_advertised() -> bool:
    """True when operators advertise x402 (default: on with pay).

    Set ``SEISO_PAY_X402=0`` to hide the rail from discovery/funding hints.
    """
    raw = (os.environ.get("SEISO_PAY_X402") or "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def x402_sim_enabled() -> bool:
    """Simulated x402 mint/verify (no chain / facilitator). Explicit or with faucet."""
    raw = (os.environ.get("SEISO_PAY_X402_SIM") or "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return faucet_enabled()


def x402_ready() -> tuple[bool, str]:
    """Whether fund/exchange may mint and settle (sim only until EVM wired)."""
    if not x402_advertised():
        return False, "SEISO_PAY_X402=0 — x402 rail hidden"
    if x402_sim_enabled():
        return True, "sim"
    return False, LIVE_NOT_READY_MSG


def require_x402_ready() -> str:
    ok, detail = x402_ready()
    if not ok:
        raise RuntimeError(detail)
    return detail


def is_evm_address(raw: str | None) -> bool:
    return bool(_EVM_RE.fullmatch((raw or "").strip()))


def normalize_evm_address(raw: str) -> str:
    text = (raw or "").strip()
    if not is_evm_address(text):
        raise ValueError(f"invalid EVM address: {raw!r}")
    return "0x" + text[2:].lower()


def list_supported_networks() -> list[dict[str, str | bool]]:
    """Return all supported networks with chain info."""
    names: dict[str, str] = {
        "eip155:1": "Ethereum",
        "eip155:10": "Optimism",
        "eip155:25": "Cronos",
        "eip155:30": "Rootstock",
        "eip155:40": "Telos",
        "eip155:56": "BNB Chain",
        "eip155:100": "Gnosis",
        "eip155:137": "Polygon PoS",
        "eip155:250": "Fantom",
        "eip155:288": "Boba",
        "eip155:324": "zkSync Era",
        "eip155:1088": "Metis",
        "eip155:1101": "Polygon zkEVM",
        "eip155:1284": "Moonbeam",
        "eip155:1285": "Moonriver",
        "eip155:2222": "Kava",
        "eip155:42220": "Celo",
        "eip155:4663": "Robinhood Chain",
        "eip155:5000": "Mantle",
        "eip155:8453": "Base",
        "eip155:34443": "Mode",
        "eip155:42161": "Arbitrum One",
        "eip155:42170": "Arbitrum Nova",
        "eip155:43114": "Avalanche C-Chain",
        "eip155:534352": "Scroll",
        "eip155:59144": "Linea",
        "eip155:660279": "Xai",
        "eip155:81457": "Blast",
        "eip155:1313161554": "Aurora",
        "eip155:167000": "Taiko",
        # Testnets
        "eip155:2442": "Polygon zkEVM Cardona (test)",
        "eip155:46630": "Robinhood Chain Testnet",
        "eip155:59141": "Linea Sepolia (test)",
        "eip155:84532": "Base Sepolia (test)",
        "eip155:80002": "Polygon Amoy (test)",
        "eip155:421614": "Arbitrum Sepolia (test)",
        "eip155:534351": "Scroll Sepolia (test)",
        "eip155:11155111": "Sepolia (test)",
        "eip155:11155420": "Optimism Sepolia (test)",
    }
    out: list[dict[str, str | bool]] = []
    for caip2, addr in sorted(USDC_BY_NETWORK.items()):
        out.append(
            {
                "caip2": caip2,
                "name": names.get(caip2, caip2),
                "usdc": addr,
                "testnet": "test" in caip2 or "test" in names.get(caip2, "").lower(),
            }
        )
    return out


def x402_network() -> str:
    raw = (os.environ.get("SEISO_PAY_X402_NETWORK") or DEFAULT_NETWORK).strip()
    return raw or DEFAULT_NETWORK


def x402_asset(network: str | None = None) -> str:
    explicit = (os.environ.get("SEISO_PAY_X402_ASSET") or "").strip()
    if explicit:
        if is_evm_address(explicit):
            return normalize_evm_address(explicit)
        raise ValueError("SEISO_PAY_X402_ASSET must be an EVM address")
    net = network or x402_network()
    asset = USDC_BY_NETWORK.get(net)
    if not asset:
        raise ValueError(
            f"no default USDC for network {net!r}; "
            f"set SEISO_PAY_X402_ASSET=0x... or choose from: "
            f"{', '.join(sorted(USDC_BY_NETWORK))}"
        )
    return asset


def operator_evm() -> str:
    raw = (os.environ.get("SEISO_OPERATOR_EVM") or "").strip()
    if raw:
        return normalize_evm_address(raw)
    return _SIM_PAY_TO


def protocol_treasury_evm() -> str:
    raw = (os.environ.get("SEISO_PROTOCOL_TREASURY_EVM") or "").strip()
    if not raw:
        return ""
    return normalize_evm_address(raw)


def sats_to_usdc_atomic(amount_sats: int) -> int:
    """Map session sats → USDC atomic (6 decimals) for the x402 challenge.

    Sim identity: ``SEISO_PAY_X402_ATOMIC_PER_SAT`` (default 1). Not a market FX.
    """
    if amount_sats <= 0:
        raise ValueError("amount_sats must be > 0")
    raw = (os.environ.get("SEISO_PAY_X402_ATOMIC_PER_SAT") or "1").strip()
    try:
        per = int(raw)
    except ValueError as exc:
        raise ValueError("SEISO_PAY_X402_ATOMIC_PER_SAT must be an integer") from exc
    if per <= 0:
        raise ValueError("SEISO_PAY_X402_ATOMIC_PER_SAT must be > 0")
    return int(amount_sats) * per


def _root_key() -> bytes:
    explicit = (os.environ.get("SEISO_PAY_X402_ROOT_KEY") or "").strip()
    if explicit:
        return hashlib.sha256(explicit.encode("utf-8")).digest()
    seed = (os.environ.get("SEISO_DATA_DIR") or "seiso-default") + "|x402-evm-v1"
    return hashlib.sha256(seed.encode("utf-8")).digest()


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(text: str) -> bytes:
    pad = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text + pad)


def encode_payment_header(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _b64url(canonical)


def decode_payment_header(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if not text:
        raise ValueError("empty payment header")
    if text.startswith("{"):
        body = json.loads(text)
    else:
        body = json.loads(_b64url_decode(text).decode("utf-8"))
    if not isinstance(body, dict):
        raise ValueError("payment header must decode to an object")
    return cast(dict[str, Any], body)


def _challenges_dir(data_dir: Path | None = None) -> Path:
    path = safe_join(pay_root(data_dir), "x402")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _challenge_path(challenge_id: str, data_dir: Path | None = None) -> Path:
    return safe_join(_challenges_dir(data_dir), f"{challenge_id}.json")


def _sign_payload(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hmac.new(_root_key(), canonical, hashlib.sha256).hexdigest()


def payment_required_header(required: dict[str, Any]) -> str:
    return encode_payment_header(required)


def www_authenticate_header(*, network: str, pay_to: str) -> str:
    return f'X402 scheme="{DEFAULT_SCHEME}", network="{network}", payTo="{pay_to}"'


def parse_payment_signature(header: str | None) -> dict[str, Any]:
    """Parse ``PAYMENT-SIGNATURE`` (base64 JSON or raw JSON)."""
    if not header:
        raise ValueError("PAYMENT-SIGNATURE header required")
    return decode_payment_header(header)


def _accepts_block(
    *,
    amount_atomic: int,
    session_id: str,
    challenge_id: str,
    network: str | None = None,
) -> dict[str, Any]:
    net = network or x402_network()
    asset = x402_asset(net)
    pay_to = operator_evm()
    return {
        "scheme": DEFAULT_SCHEME,
        "network": net,
        "maxAmountRequired": str(amount_atomic),
        "resource": f"seiso://pay/session/{session_id}/fund/{challenge_id}",
        "description": "Seiso marketplace session top-up (prepaid compute)",
        "mimeType": "application/json",
        "payTo": pay_to,
        "maxTimeoutSeconds": _DEFAULT_TTL_S,
        "asset": asset,
        "extra": {
            "name": "USD Coin",
            "version": "2",
            "decimals": 6,
        },
    }


def mint_fund_challenge(
    *,
    session_id: str,
    amount_sats: int,
    data_dir: Path | None = None,
    ttl_s: int | None = None,
    network: str | None = None,
) -> dict[str, Any]:
    """Mint an x402 EVM challenge for session top-up (sim mode).

    Args:
        session_id: The prepaid session to fund.
        amount_sats: Amount in sats to credit.
        data_dir: Override data directory.
        ttl_s: Challenge TTL in seconds.
        network: CAIP-2 network override (default from env/SEISO_PAY_X402_NETWORK).
    """
    if amount_sats <= 0:
        raise ValueError("amount_sats must be > 0")
    mode = require_x402_ready()
    ttl = int(ttl_s or os.environ.get("SEISO_PAY_X402_TTL_S") or _DEFAULT_TTL_S)
    challenge_id = uuid.uuid4().hex
    amount_atomic = sats_to_usdc_atomic(amount_sats)
    now = time.time()
    exp = now + max(60, ttl)
    nonce = "0x" + secrets.token_hex(32)
    net = network or x402_network()
    accepts = _accepts_block(
        amount_atomic=amount_atomic,
        session_id=session_id,
        challenge_id=challenge_id,
        network=net,
    )
    required = {
        "x402Version": X402_VERSION,
        "error": "PAYMENT_REQUIRED",
        "accepts": [accepts],
    }
    sim_sig = _sign_payload(
        {
            "challenge_id": challenge_id,
            "session_id": session_id,
            "amount_sats": int(amount_sats),
            "amount_atomic": amount_atomic,
            "nonce": nonce,
        }
    )
    record: dict[str, Any] = {
        "challenge_id": challenge_id,
        "session_id": session_id,
        "amount_sats": int(amount_sats),
        "amount_atomic": amount_atomic,
        "network": accepts["network"],
        "asset": accepts["asset"],
        "pay_to": accepts["payTo"],
        "scheme": DEFAULT_SCHEME,
        "mode": mode,
        "status": "pending",
        "created_at": now,
        "expires_at": exp,
        "nonce": nonce,
        "payment_required": required,
        "sim_signature": sim_sig,
    }
    path = _challenge_path(challenge_id, data_dir)
    with _lock:
        path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        append_ledger(
            {
                "event": "x402_challenge_mint",
                "challenge_id": challenge_id,
                "session_id": session_id,
                "amount_sats": amount_sats,
                "amount_atomic": amount_atomic,
                "network": net,
                "mode": mode,
                "ts": now,
            },
            data_dir=data_dir,
        )
    return public_challenge_view(record, include_sim_signature=True)


def public_challenge_view(
    record: dict[str, Any],
    *,
    include_sim_signature: bool = False,
) -> dict[str, Any]:
    required = dict(record["payment_required"])
    header = payment_required_header(required)
    network = str(record["network"])
    pay_to = str(record["pay_to"])
    amount = int(record["amount_sats"])
    out: dict[str, Any] = {
        "method": "x402",
        "status": "ready" if record.get("mode") == "sim" else "not_functional",
        "mode": record.get("mode"),
        "challenge_id": record.get("challenge_id"),
        "session_id": record.get("session_id"),
        "amount_sats": amount,
        "amount_usdc_atomic": int(record["amount_atomic"]),
        "scheme": DEFAULT_SCHEME,
        "network": network,
        "asset": record.get("asset"),
        "pay_to": pay_to,
        "http_status": 402,
        "x402Version": X402_VERSION,
        "payment_required": required,
        "payment_required_header": header,
        "www_authenticate": www_authenticate_header(network=network, pay_to=pay_to),
        "expires_at": record.get("expires_at"),
        "reference": REFERENCE_URL,
        "site": SITE_URL,
        "do_not_use_live_evm": True,
        "detail": (
            "Simulated x402 EVM challenge — complete with PAYMENT-SIGNATURE "
            "(sim_payment_signature / CLI --x402). Live USDC / facilitator not wired."
            if record.get("mode") == "sim"
            else LIVE_NOT_READY_MSG
        ),
    }
    if include_sim_signature and record.get("mode") == "sim":
        payload = sim_payment_signature_payload(record)
        out["sim_payment_signature"] = encode_payment_header(payload)
        out["sim_payment_payload"] = payload
        out["sim_hint"] = (
            "Dev only: retry with header PAYMENT-SIGNATURE: "
            "<sim_payment_signature> then POST /pay/v1/sessions/fund/x402/complete"
        )
    return out


def sim_payment_signature_payload(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "x402Version": X402_VERSION,
        "scheme": DEFAULT_SCHEME,
        "network": record["network"],
        "payload": {
            "signature": "0x" + str(record["sim_signature"]),
            "authorization": {
                "from": _SIM_PAY_TO,
                "to": record["pay_to"],
                "value": str(record["amount_atomic"]),
                "validAfter": "0",
                "validBefore": str(int(float(record["expires_at"]))),
                "nonce": record["nonce"],
            },
            "challenge_id": record["challenge_id"],
        },
    }


def load_challenge(challenge_id: str, data_dir: Path | None = None) -> dict[str, Any]:
    path = _challenge_path(challenge_id, data_dir)
    if not path.is_file():
        raise KeyError(f"x402 challenge not found: {challenge_id}")
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _save_challenge(record: dict[str, Any], data_dir: Path | None = None) -> None:
    path = _challenge_path(str(record["challenge_id"]), data_dir)
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def _challenge_id_from_payload(payload: dict[str, Any]) -> str:
    inner = payload.get("payload")
    if isinstance(inner, dict) and inner.get("challenge_id"):
        return str(inner["challenge_id"])
    extra = payload.get("challenge_id")
    if extra:
        return str(extra)
    raise ValueError("PAYMENT-SIGNATURE missing challenge_id")


def complete_fund(
    *,
    payment_signature: str | None = None,
    payload: dict[str, Any] | None = None,
    data_dir: Path | None = None,
) -> dict[str, Any]:
    """Verify an x402 PAYMENT-SIGNATURE and credit the session (funding_mode=x402)."""
    require_x402_ready()
    if payment_signature and payload is None:
        payload = parse_payment_signature(payment_signature)
    if not payload:
        raise ValueError("PAYMENT-SIGNATURE or payload required")

    challenge_id = _challenge_id_from_payload(payload)
    with _lock:
        record = load_challenge(challenge_id, data_dir)
        if record.get("status") == "settled":
            raise RuntimeError("x402 challenge already settled")
        if float(record.get("expires_at") or 0) < time.time():
            raise RuntimeError("x402 challenge expired")
        if str(payload.get("scheme") or DEFAULT_SCHEME) != DEFAULT_SCHEME:
            raise ValueError("only the x402 exact scheme is supported")
        if str(payload.get("network") or "") != str(record["network"]):
            raise ValueError("PAYMENT-SIGNATURE network mismatch")

        inner = payload.get("payload")
        if not isinstance(inner, dict):
            raise ValueError("PAYMENT-SIGNATURE payload object required")
        auth = inner.get("authorization")
        if not isinstance(auth, dict):
            raise ValueError("PAYMENT-SIGNATURE authorization required")
        if str(auth.get("nonce") or "") != str(record["nonce"]):
            raise ValueError("authorization nonce mismatch")
        if str(auth.get("value") or "") != str(record["amount_atomic"]):
            raise ValueError("authorization value must exactly match maxAmountRequired")
        if normalize_evm_address(str(auth.get("to") or "")) != normalize_evm_address(
            str(record["pay_to"])
        ):
            raise ValueError("authorization.to must match payTo")

        expected = _sign_payload(
            {
                "challenge_id": challenge_id,
                "session_id": record["session_id"],
                "amount_sats": int(record["amount_sats"]),
                "amount_atomic": int(record["amount_atomic"]),
                "nonce": record["nonce"],
            }
        )
        sig = str(inner.get("signature") or "").strip()
        if sig.startswith("0x"):
            sig = sig[2:]
        if not hmac.compare_digest(sig, expected):
            raise ValueError("invalid x402 payment signature")

        amount = int(record["amount_sats"])
        session = activate_session(
            str(record["session_id"]),
            amount_sats=amount,
            data_dir=data_dir,
            funding_mode="x402",
            fund_id=challenge_id,
        )
        record["status"] = "settled"
        record["settled_at"] = time.time()
        record.pop("sim_signature", None)
        _save_challenge(record, data_dir)
        append_ledger(
            {
                "event": "x402_fund_complete",
                "challenge_id": challenge_id,
                "session_id": record["session_id"],
                "amount_sats": amount,
                "amount_atomic": record["amount_atomic"],
                "network": record["network"],
                "balance_sats": session.get("balance_sats"),
                "ts": time.time(),
            },
            data_dir=data_dir,
        )

    from seiso.pay.store import public_session_view

    return {
        "session": public_session_view(session),
        "challenge_id": challenge_id,
        "amount_sats": amount,
        "amount_usdc_atomic": int(record["amount_atomic"]),
        "funding_mode": "x402",
        "network": record["network"],
        "mode": record.get("mode"),
    }


def challenge_placeholder(*, session_id: str, amount_sats: int) -> dict[str, Any]:
    """Discovery-only block when x402 is advertised but not ready to mint."""
    network = x402_network()
    return {
        "method": "x402",
        "status": "not_functional",
        "do_not_use": True,
        "do_not_use_live_evm": True,
        "session_id": session_id,
        "amount_sats": int(amount_sats),
        "http_status": 402,
        "scheme": DEFAULT_SCHEME,
        "network": network,
        "www_authenticate": None,
        "payment_required_header": None,
        "hint": (
            "When wired or with SEISO_PAY_X402_SIM=1: client reads "
            "PAYMENT-REQUIRED, signs an EIP-3009 exact transfer, retries with "
            "PAYMENT-SIGNATURE"
        ),
        "reference": REFERENCE_URL,
        "site": SITE_URL,
        "detail": LIVE_NOT_READY_MSG,
    }


def funding_x402_block(session_id: str, amount_sats: int) -> dict[str, Any] | None:
    """x402 slice of session funding instructions, or None when not advertised."""
    if not x402_advertised():
        return None
    ok, _ = x402_ready()
    if ok and amount_sats > 0:
        return {
            "method": "x402",
            "status": "ready",
            "mode": "sim",
            "session_id": session_id,
            "amount_sats": int(amount_sats),
            "scheme": DEFAULT_SCHEME,
            "network": x402_network(),
            "do_not_use_live_evm": True,
            "endpoints": {
                "challenge": "POST /pay/v1/sessions/fund/x402 (Bearer required)",
                "complete": "POST /pay/v1/sessions/fund/x402/complete",
            },
            "headers": {
                "challenge": "PAYMENT-REQUIRED",
                "complete": "PAYMENT-SIGNATURE",
            },
            "cli": "seiso pay session fund --session ID --sats N --x402",
            "reference": REFERENCE_URL,
            "site": SITE_URL,
            "detail": (
                "Simulated x402 EVM fund/exchange available. Live USDC / "
                "facilitator not wired — do not use for real funds."
            ),
        }
    block = challenge_placeholder(session_id=session_id, amount_sats=amount_sats)
    if faucet_enabled():
        block["dev_note"] = (
            "Dev faucet is enabled; enable SEISO_PAY_X402_SIM=1 for x402 sim "
            "fund/exchange, or use --faucet."
        )
    return block
