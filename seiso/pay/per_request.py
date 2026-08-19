"""Per-request payment challenges (x402 USDC, ETH router, L402, Ark).

One HTTP request → one quote → one 402 with every advertised rail → one
receipt. Prepaid sessions remain optional; they are not required for
``/v1/chat/completions`` when a per-request proof is presented.

Supports all EVM chains via the x402 rail (``SEISO_PAY_X402_NETWORK``).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, cast

from seiso.pay.flags import payment_methods
from seiso.pay.fx import quote_fx
from seiso.pay.l402 import funding_l402_block, l402_sim_enabled
from seiso.pay.oracle import OraclePrices, load_prices
from seiso.pay.pricing import quote_inference_tokens
from seiso.pay.store import append_ledger, pay_root
from seiso.pay.x402 import (
    DEFAULT_NETWORK,
    DEFAULT_SCHEME,
    X402_VERSION,
    encode_payment_header,
    operator_evm,
    x402_advertised,
    x402_asset,
    x402_network,
    x402_sim_enabled,
)
from seiso.security import safe_join

_REQUESTS = "requests"


def per_request_enabled() -> bool:
    raw = (os.environ.get("SEISO_PAY_PER_REQUEST") or "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _requests_dir(data_dir: Path | None = None) -> Path:
    path = safe_join(pay_root(data_dir), _REQUESTS)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _request_path(request_id: str, data_dir: Path | None = None) -> Path:
    return safe_join(_requests_dir(data_dir), f"{request_id}.json")


def save_request(record: dict[str, Any], data_dir: Path | None = None) -> None:
    path = _request_path(str(record["request_id"]), data_dir)
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def load_request(request_id: str, data_dir: Path | None = None) -> dict[str, Any]:
    path = _request_path(request_id, data_dir)
    if not path.is_file():
        raise KeyError(f"pay request not found: {request_id}")
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def request_paid(request_id: str, data_dir: Path | None = None) -> bool:
    try:
        rec = load_request(request_id, data_dir)
    except KeyError:
        return False
    return rec.get("status") == "paid"


def mint_request_quote(
    *,
    prompt_tokens: int,
    completion_tokens: int,
    data_dir: Path | None = None,
    prices: OraclePrices | None = None,
    flat_call: bool = False,
) -> dict[str, Any]:
    """Build a per-request quote + 402 payment options with all rails."""
    token_quote = quote_inference_tokens(prompt_tokens, completion_tokens, flat_call=flat_call)
    prices = prices or load_prices()
    fx = quote_fx(
        int(token_quote["compute_sats"]),
        btc_usd_8=prices.btc_usd_8,
        eth_usd_8=prices.eth_usd_8,
        bps=int(token_quote["protocol_fee_bps"]),
    )
    request_id = uuid.uuid4().hex
    now = time.time()
    deadline = now + 600
    record: dict[str, Any] = {
        "request_id": request_id,
        "status": "pending",
        "created_at": now,
        "deadline": deadline,
        "token_quote": token_quote,
        "fx": fx.as_dict(),
        "paid_via": None,
    }
    save_request(record, data_dir)

    net = x402_network()
    rails: dict[str, Any] = {}

    # x402 EVM (USDC on any supported chain)
    if x402_advertised():
        rails["x402"] = {
            "method": "x402",
            "scheme": DEFAULT_SCHEME,
            "network": net,
            "asset": x402_asset(),
            "pay_to": operator_evm(),
            "max_amount_required": str(fx.usdc_atomic),
            "amount_usdc_atomic": fx.usdc_atomic,
            "sim": x402_sim_enabled(),
            "endpoints": {
                "challenge": "POST /pay/v1/requests/{id}/x402",
                "complete": "POST /pay/v1/requests/{id}/x402/complete",
            },
        }

    # ETH native via SeisoPayRouter
    rails["eth"] = {
        "method": "eth",
        "asset": "native",
        "wei": fx.wei,
        "router": (os.environ.get("SEISO_PAY_ROUTER") or "").strip() or None,
        "oracle": prices.as_dict(),
        "function": "payETH(Quote,bytes)",
        "detail": (
            "Send Quote-signed payETH on SeisoPayRouter. "
            "msg.value must be >= wei; surplus is refunded."
        ),
    }

    # L402 Lightning
    rails["l402"] = funding_l402_block(request_id, int(fx.total_sats))

    # Ark
    from seiso.pay.ark import funding_instructions

    ark = funding_instructions(request_id, int(fx.total_sats))
    rails["ark"] = {
        "method": "ark",
        "ark_address": ark.get("ark_address"),
        "amount_sats": fx.total_sats,
        "status": "per_request",
        "do_not_use_live_rails": True,
    }

    # BTC HTLC
    op_pub = (os.environ.get("SEISO_PAY_BTC_OPERATOR_PUBKEY") or "").strip()
    by_pub = (os.environ.get("SEISO_PAY_BTC_BUYER_PUBKEY") or "").strip()
    if op_pub and by_pub:
        from seiso.pay.btc.htlc import build_htlc, payment_hash_from_preimage

        preimage = hashlib.sha256(request_id.encode("utf-8")).digest()
        offer = build_htlc(
            payment_hash=payment_hash_from_preimage(preimage),
            operator_pubkey=bytes.fromhex(op_pub),
            buyer_pubkey=bytes.fromhex(by_pub),
            locktime=int(deadline),
            amount_sats=int(fx.total_sats),
        )
        rails["btc_htlc"] = offer.as_dict()
        rails["btc_htlc"]["method"] = "btc_htlc"

    required = {
        "x402Version": X402_VERSION,
        "error": "PAYMENT_REQUIRED",
        "request_id": request_id,
        "accepts": [
            {
                "scheme": DEFAULT_SCHEME,
                "network": net if x402_advertised() else DEFAULT_NETWORK,
                "maxAmountRequired": str(fx.usdc_atomic),
                "payTo": operator_evm(),
                "asset": x402_asset() if x402_advertised() else None,
                "extra": {"asset": "USDC", "decimals": 6},
            },
            {
                "scheme": "exact",
                "network": net if x402_advertised() else DEFAULT_NETWORK,
                "maxAmountRequired": str(fx.wei),
                "payTo": (os.environ.get("SEISO_PAY_ROUTER") or operator_evm()),
                "asset": "native",
                "extra": {"symbol": "ETH", "decimals": 18, "wei": fx.wei},
            },
        ],
    }
    header = encode_payment_header(required)
    challenge = {
        "http_status": 402,
        "request_id": request_id,
        "quote": token_quote,
        "fx": fx.as_dict(),
        "rails": rails,
        "payment_methods": payment_methods(),
        "payment_required": required,
        "payment_required_header": header,
        "www_authenticate": (
            f'X402 scheme="{DEFAULT_SCHEME}", ETH wei={fx.wei}, L402 sats={fx.total_sats}'
        ),
        "deadline": deadline,
        "do_not_use_live_rails": True,
        "per_request": True,
    }
    record["challenge"] = {k: challenge[k] for k in ("fx", "deadline")}
    save_request(record, data_dir)
    append_ledger(
        {
            "event": "per_request_quote",
            "request_id": request_id,
            "total_sats": fx.total_sats,
            "wei": fx.wei,
            "usdc_atomic": fx.usdc_atomic,
            "network": net,
            "ts": now,
        },
        data_dir=data_dir,
    )
    return challenge


def _root_key() -> bytes:
    explicit = (os.environ.get("SEISO_PAY_X402_ROOT_KEY") or "").strip()
    if explicit:
        return hashlib.sha256(explicit.encode("utf-8")).digest()
    seed = (os.environ.get("SEISO_DATA_DIR") or "seiso-default") + "|per-request-v1"
    return hashlib.sha256(seed.encode("utf-8")).digest()


def sim_receipt(request_id: str, *, via: str, data_dir: Path | None = None) -> str:
    """HMAC receipt for sim rails (never a chain proof)."""
    rec = load_request(request_id, data_dir)
    payload = {
        "request_id": request_id,
        "via": via,
        "total_sats": rec["fx"]["total_sats"],
        "wei": rec["fx"]["wei"],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hmac.new(_root_key(), canonical, hashlib.sha256).hexdigest()


def mark_paid(
    request_id: str,
    *,
    via: str,
    proof: str,
    data_dir: Path | None = None,
) -> dict[str, Any]:
    rec = load_request(request_id, data_dir)
    if rec.get("status") == "paid":
        if rec.get("proof") == proof:
            return rec
        raise RuntimeError("request already paid")
    if rec.get("deadline") and float(rec["deadline"]) < time.time():
        raise RuntimeError("request quote expired")
    rec["status"] = "paid"
    rec["paid_via"] = via
    rec["proof"] = proof
    rec["paid_at"] = time.time()
    save_request(rec, data_dir)
    append_ledger(
        {
            "event": "per_request_paid",
            "request_id": request_id,
            "via": via,
            "ts": rec["paid_at"],
        },
        data_dir=data_dir,
    )
    return rec


def complete_sim(
    request_id: str,
    *,
    via: str,
    receipt: str,
    data_dir: Path | None = None,
) -> dict[str, Any]:
    rec = load_request(request_id, data_dir)
    if rec.get("status") == "paid":
        if rec.get("proof") == receipt:
            return rec
        raise RuntimeError("request already paid")
    expected = sim_receipt(request_id, via=via, data_dir=data_dir)
    if not hmac.compare_digest(receipt, expected):
        raise ValueError("invalid per-request receipt")
    return mark_paid(request_id, via=via, proof=receipt, data_dir=data_dir)


def complete_x402_request(
    request_id: str,
    *,
    payment_signature: str | None = None,
    receipt: str | None = None,
    data_dir: Path | None = None,
) -> dict[str, Any]:
    """Mark a request paid via x402 sim receipt (no prepaid session).

    Supports all EVM chains configured in ``SEISO_PAY_X402_NETWORK``.
    """
    if not x402_sim_enabled():
        raise RuntimeError(
            "Live x402 per-request settle is not wired. "
            "Set SEISO_PAY_X402_SIM=1 or payETH on SeisoPayRouter."
        )
    rec = load_request(request_id, data_dir)
    if rec.get("status") == "paid":
        return rec
    _ = payment_signature  # live path will verify EIP-3009 / facilitator
    proof = (receipt or sim_receipt(request_id, via="x402", data_dir=data_dir)).strip()
    return complete_sim(request_id, via="x402", receipt=proof, data_dir=data_dir)


def complete_l402_request(
    request_id: str,
    *,
    receipt: str | None = None,
    data_dir: Path | None = None,
) -> dict[str, Any]:
    if not l402_sim_enabled():
        raise RuntimeError("Live L402 per-request settle is not wired. Set SEISO_PAY_L402_SIM=1.")
    rec = load_request(request_id, data_dir)
    if rec.get("status") == "paid":
        return rec
    proof = (receipt or sim_receipt(request_id, via="l402", data_dir=data_dir)).strip()
    return complete_sim(request_id, via="l402", receipt=proof, data_dir=data_dir)


def complete_eth_request(
    request_id: str,
    *,
    receipt: str | None = None,
    data_dir: Path | None = None,
) -> dict[str, Any]:
    """Sim ETH payETH. Live path: watch SeisoPayRouter.RequestPaid."""
    rec = load_request(request_id, data_dir)
    if rec.get("status") == "paid":
        return rec
    sim = (os.environ.get("SEISO_PAY_X402_SIM") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    from seiso.pay.flags import faucet_enabled

    if not (sim or faucet_enabled()):
        raise RuntimeError(
            "Live ETH per-request settle requires SeisoPayRouter.RequestPaid. "
            "Set SEISO_PAY_X402_SIM=1 for simulated receipts."
        )
    proof = (receipt or sim_receipt(request_id, via="eth", data_dir=data_dir)).strip()
    return complete_sim(request_id, via="eth", receipt=proof, data_dir=data_dir)
