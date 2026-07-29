"""L402 (Lightning HTTP 402) funding rail for prepaid marketplace sessions.

L402 joins HTTP 402 Payment Required with a Lightning invoice + macaroon
challenge so clients (including agents) can pay sats and exchange once for a
``seiso_pay_*`` session token. Later jobs/inference use Bearer debit — not
per-request L402.

Reference: https://lightningfaucet.com/learn/l402-payments-explained/

**Live Lightning** (real BOLT-11 / node settle) is **not functional yet — do
not use for real funds.** Set ``SEISO_PAY_L402_SIM=1`` (or enable the faucet)
for end-to-end simulated challenges that credit session balance. Job failures
refund to the prepaid session balance (Lightning payments are one-way).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
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

LIVE_NOT_READY_MSG = (
    "Live L402 Lightning settlement is not functional yet — do not use for "
    "real funds. Set SEISO_PAY_L402_SIM=1 (or SEISO_PAY_FAUCET=1) for simulated "
    "fund/exchange smoke tests. "
    "See https://lightningfaucet.com/learn/l402-payments-explained/"
)

REFERENCE_URL = "https://lightningfaucet.com/learn/l402-payments-explained/"

# Challenge lifetime (seconds)
_DEFAULT_TTL_S = 600


def l402_advertised() -> bool:
    """True when operators opt into advertising L402 (default: on with pay).

    Set ``SEISO_PAY_L402=0`` to hide the rail from discovery/funding hints.
    """
    raw = (os.environ.get("SEISO_PAY_L402") or "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def l402_sim_enabled() -> bool:
    """Simulated L402 mint/verify (no Lightning node). Explicit or with faucet."""
    raw = (os.environ.get("SEISO_PAY_L402_SIM") or "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    # Default: allow sim when faucet is on (local smoke), else off.
    return faucet_enabled()


def l402_ready() -> tuple[bool, str]:
    """Whether fund/exchange may mint and settle (sim only until LN wired)."""
    if not l402_advertised():
        return False, "SEISO_PAY_L402=0 — L402 rail hidden"
    if l402_sim_enabled():
        return True, "sim"
    return False, LIVE_NOT_READY_MSG


def require_l402_ready() -> str:
    """Return mode (``sim``) or raise if live settle attempted without wire."""
    ok, detail = l402_ready()
    if not ok:
        raise RuntimeError(detail)
    return detail


def _root_key() -> bytes:
    """HMAC root for macaroon signatures (dev-derived; not a Lightning secret)."""
    explicit = (os.environ.get("SEISO_PAY_L402_ROOT_KEY") or "").strip()
    if explicit:
        return hashlib.sha256(explicit.encode("utf-8")).digest()
    # Stable-per-data-dir derived key so restarts accept in-flight challenges.
    seed = (os.environ.get("SEISO_DATA_DIR") or "seiso-default") + "|l402-macaroon-v1"
    return hashlib.sha256(seed.encode("utf-8")).digest()


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(text: str) -> bytes:
    pad = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text + pad)


def _challenges_dir(data_dir: Path | None = None) -> Path:
    path = safe_join(pay_root(data_dir), "l402")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _challenge_path(challenge_id: str, data_dir: Path | None = None) -> Path:
    return safe_join(_challenges_dir(data_dir), f"{challenge_id}.json")


def _sign_payload(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hmac.new(_root_key(), canonical, hashlib.sha256).hexdigest()


def _mint_macaroon(
    *,
    challenge_id: str,
    session_id: str,
    amount_sats: int,
    payment_hash: str,
    exp: float,
) -> str:
    body = {
        "v": 1,
        "challenge_id": challenge_id,
        "session_id": session_id,
        "amount_sats": int(amount_sats),
        "payment_hash": payment_hash,
        "exp": exp,
        "caveats": ["session_fund", f"amount={int(amount_sats)}"],
    }
    body["sig"] = _sign_payload({k: v for k, v in body.items() if k != "sig"})
    return _b64url(json.dumps(body, separators=(",", ":")).encode("utf-8"))


def _parse_macaroon(macaroon_b64: str) -> dict[str, Any]:
    raw = _b64url_decode(macaroon_b64.strip())
    body = cast(dict[str, Any], json.loads(raw.decode("utf-8")))
    sig = str(body.get("sig") or "")
    check = {k: v for k, v in body.items() if k != "sig"}
    expected = _sign_payload(check)
    if not hmac.compare_digest(sig, expected):
        raise ValueError("invalid macaroon signature")
    if float(body.get("exp") or 0) < time.time():
        raise ValueError("macaroon expired")
    return body


def www_authenticate_header(*, macaroon: str, invoice: str) -> str:
    return f'L402 macaroon="{macaroon}", invoice="{invoice}"'


def parse_l402_authorization(header: str | None) -> tuple[str, str]:
    """Parse ``Authorization: L402 <macaroon>:<preimage_hex>``."""
    if not header:
        raise ValueError("Authorization header required")
    parts = header.strip().split(None, 1)
    if len(parts) != 2 or parts[0].upper() != "L402":
        raise ValueError("expected Authorization: L402 <macaroon>:<preimage>")
    cred = parts[1].strip()
    if ":" not in cred:
        raise ValueError("L402 credentials must be macaroon:preimage")
    macaroon, preimage = cred.split(":", 1)
    macaroon = macaroon.strip().strip('"')
    preimage = preimage.strip().lower()
    if not macaroon or not preimage:
        raise ValueError("empty macaroon or preimage")
    return macaroon, preimage


def _sim_invoice(payment_hash: str, amount_sats: int) -> str:
    """Non-payable placeholder invoice encoding hash + amount (sim only)."""
    return f"lnbcsseisosim1{payment_hash[:40]}amt{int(amount_sats)}"


def mint_fund_challenge(
    *,
    session_id: str,
    amount_sats: int,
    data_dir: Path | None = None,
    ttl_s: int | None = None,
) -> dict[str, Any]:
    """Mint an L402 challenge for session top-up (sim mode).

    Returns challenge fields plus ``www_authenticate``. In sim mode also
    includes ``sim_preimage`` so CLI can complete without a Lightning wallet.
    Live LN minting is not wired — raises via ``require_l402_ready``.
    """
    if amount_sats <= 0:
        raise ValueError("amount_sats must be > 0")
    mode = require_l402_ready()
    ttl = int(ttl_s or os.environ.get("SEISO_PAY_L402_TTL_S") or _DEFAULT_TTL_S)
    challenge_id = uuid.uuid4().hex
    preimage = secrets.token_bytes(32)
    preimage_hex = preimage.hex()
    payment_hash = hashlib.sha256(preimage).hexdigest()
    now = time.time()
    exp = now + max(60, ttl)
    invoice = _sim_invoice(payment_hash, amount_sats)
    macaroon = _mint_macaroon(
        challenge_id=challenge_id,
        session_id=session_id,
        amount_sats=amount_sats,
        payment_hash=payment_hash,
        exp=exp,
    )
    record: dict[str, Any] = {
        "challenge_id": challenge_id,
        "session_id": session_id,
        "amount_sats": int(amount_sats),
        "payment_hash": payment_hash,
        "invoice": invoice,
        "macaroon": macaroon,
        "mode": mode,
        "status": "pending",
        "created_at": now,
        "expires_at": exp,
        # Stored for sim complete; never returned on public funding discovery.
        "sim_preimage": preimage_hex,
    }
    path = _challenge_path(challenge_id, data_dir)
    with _lock:
        path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        append_ledger(
            {
                "event": "l402_challenge_mint",
                "challenge_id": challenge_id,
                "session_id": session_id,
                "amount_sats": amount_sats,
                "mode": mode,
                "ts": now,
            },
            data_dir=data_dir,
        )
    return public_challenge_view(record, include_sim_preimage=True)


def public_challenge_view(
    record: dict[str, Any],
    *,
    include_sim_preimage: bool = False,
) -> dict[str, Any]:
    macaroon = str(record["macaroon"])
    invoice = str(record["invoice"])
    amount = int(record["amount_sats"])
    out: dict[str, Any] = {
        "method": "l402",
        "status": "ready" if record.get("mode") == "sim" else "not_functional",
        "mode": record.get("mode"),
        "challenge_id": record.get("challenge_id"),
        "session_id": record.get("session_id"),
        "amount_sats": amount,
        "price_msat": amount * 1000,
        "compute_sats": amount,  # full top-up credits buyer balance
        "protocol_fee_sats": 0,  # fee taken at spend time, not fund
        "total_sats": amount,
        "http_status": 402,
        "macaroon": macaroon,
        "invoice": invoice,
        "payment_hash": record.get("payment_hash"),
        "www_authenticate": www_authenticate_header(macaroon=macaroon, invoice=invoice),
        "expires_at": record.get("expires_at"),
        "reference": REFERENCE_URL,
        "do_not_use_live_ln": True,
        "detail": (
            "Simulated L402 challenge — pay via sim_preimage / CLI --l402, "
            "not a real Lightning invoice. Live LN not wired."
            if record.get("mode") == "sim"
            else LIVE_NOT_READY_MSG
        ),
    }
    if include_sim_preimage and record.get("mode") == "sim":
        out["sim_preimage"] = record.get("sim_preimage")
        out["sim_hint"] = (
            "Dev only: Authorization: L402 <macaroon>:<sim_preimage> "
            "then POST /pay/v1/sessions/fund/l402/complete"
        )
    return out


def load_challenge(challenge_id: str, data_dir: Path | None = None) -> dict[str, Any]:
    path = _challenge_path(challenge_id, data_dir)
    if not path.is_file():
        raise KeyError(f"l402 challenge not found: {challenge_id}")
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _save_challenge(record: dict[str, Any], data_dir: Path | None = None) -> None:
    path = _challenge_path(str(record["challenge_id"]), data_dir)
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def complete_fund(
    *,
    authorization: str | None = None,
    macaroon: str | None = None,
    preimage_hex: str | None = None,
    data_dir: Path | None = None,
) -> dict[str, Any]:
    """Verify L402 credentials and credit the session (funding_mode=l402)."""
    require_l402_ready()
    if authorization:
        macaroon, preimage_hex = parse_l402_authorization(authorization)
    if not macaroon or not preimage_hex:
        raise ValueError("macaroon and preimage required")
    preimage_hex = preimage_hex.strip().lower()
    try:
        preimage = bytes.fromhex(preimage_hex)
    except ValueError as exc:
        raise ValueError("preimage must be hex") from exc
    if len(preimage) != 32:
        raise ValueError("preimage must be 32 bytes")

    body = _parse_macaroon(macaroon)
    challenge_id = str(body["challenge_id"])
    payment_hash = hashlib.sha256(preimage).hexdigest()
    if not hmac.compare_digest(payment_hash, str(body["payment_hash"])):
        raise ValueError("preimage does not match payment hash")

    with _lock:
        record = load_challenge(challenge_id, data_dir)
        if record.get("status") == "settled":
            raise RuntimeError("l402 challenge already settled")
        if float(record.get("expires_at") or 0) < time.time():
            raise RuntimeError("l402 challenge expired")
        if str(record.get("session_id")) != str(body["session_id"]):
            raise ValueError("macaroon session mismatch")
        if int(record.get("amount_sats") or 0) != int(body["amount_sats"]):
            raise ValueError("macaroon amount mismatch")
        if not hmac.compare_digest(str(record["payment_hash"]), payment_hash):
            raise ValueError("payment hash mismatch")

        amount = int(record["amount_sats"])
        session = activate_session(
            str(record["session_id"]),
            amount_sats=amount,
            data_dir=data_dir,
            funding_mode="l402",
        )
        record["status"] = "settled"
        record["settled_at"] = time.time()
        record.pop("sim_preimage", None)
        _save_challenge(record, data_dir)
        append_ledger(
            {
                "event": "l402_fund_complete",
                "challenge_id": challenge_id,
                "session_id": record["session_id"],
                "amount_sats": amount,
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
        "funding_mode": "l402",
        "mode": record.get("mode"),
    }


def challenge_placeholder(
    *,
    session_id: str,
    amount_sats: int,
) -> dict[str, Any]:
    """Discovery-only block when L402 is advertised but not ready to mint."""
    return {
        "method": "l402",
        "status": "not_functional",
        "do_not_use": True,
        "session_id": session_id,
        "amount_sats": int(amount_sats),
        "http_status": 402,
        "www_authenticate": None,
        "macaroon": None,
        "invoice": None,
        "hint": (
            "When wired or with SEISO_PAY_L402_SIM=1: client pays BOLT-11 "
            "(or sim), retries with Authorization: L402 <macaroon>:<preimage>"
        ),
        "reference": REFERENCE_URL,
        "detail": LIVE_NOT_READY_MSG,
    }


def funding_l402_block(session_id: str, amount_sats: int) -> dict[str, Any] | None:
    """L402 slice of session funding instructions, or None when not advertised."""
    if not l402_advertised():
        return None
    ok, _ = l402_ready()
    if ok and amount_sats > 0:
        # Do not auto-mint on every funding_instructions call (would spam
        # challenges). Point clients at the fund endpoint instead.
        return {
            "method": "l402",
            "status": "ready",
            "mode": "sim",
            "session_id": session_id,
            "amount_sats": int(amount_sats),
            "do_not_use_live_ln": True,
            "endpoints": {
                "challenge": "POST /pay/v1/sessions/fund/l402",
                "complete": "POST /pay/v1/sessions/fund/l402/complete",
            },
            "cli": "seiso pay session fund --session ID --sats N --l402",
            "reference": REFERENCE_URL,
            "detail": (
                "Simulated L402 fund/exchange available. Live Lightning not "
                "wired — do not use for real funds."
            ),
        }
    block = challenge_placeholder(session_id=session_id, amount_sats=amount_sats)
    if faucet_enabled():
        block["dev_note"] = (
            "Dev faucet is enabled; enable SEISO_PAY_L402_SIM=1 for L402 sim "
            "fund/exchange, or use --faucet."
        )
    return block
