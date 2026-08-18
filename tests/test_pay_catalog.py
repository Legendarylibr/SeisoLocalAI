"""Unit tests for marketplace catalog listings (no Seiso token)."""

from __future__ import annotations

import pytest

from seiso.pay.catalog import (
    FORBIDDEN_LISTING_KEYS,
    LISTING_KINDS,
    Listing,
    live_settle_allowed,
    parse_listing_kind,
    quote_listing,
)
from seiso.pay.pricing import fee_split, quote_job


def _listing(**kwargs: object) -> Listing:
    base: dict[str, object] = dict(
        kind="finetune",
        label="QLoRA smoke",
        operator_id="op1",
        compute_sats=2500,
        gpu_class="4090",
        model_or_preset="smoke",
    )
    base.update(kwargs)
    return Listing(**base)  # type: ignore[arg-type]


def test_listing_kinds_cover_jobs() -> None:
    for kind in ("inference", "finetune", "slime", "distill_rl", "nemo_rl"):
        assert kind in LISTING_KINDS


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("finetune", "finetune"),
        ("chat", "inference"),
        ("code", "inference"),
        ("slime", "slime"),
        ("distill-rl", "distill_rl"),
    ],
)
def test_parse_listing_kind(raw: str, expected: str) -> None:
    assert parse_listing_kind(raw) == expected


def test_unknown_listing_kind() -> None:
    with pytest.raises(ValueError):
        parse_listing_kind("mining")


def test_quote_uses_fee_split() -> None:
    q = quote_listing(_listing(compute_sats=10_000), bps=500)
    split = fee_split(10_000, bps=500)
    assert q["compute_sats"] == split.compute_sats
    assert q["protocol_fee_sats"] == split.protocol_fee_sats
    assert q["total_sats"] == split.total_sats
    assert q["price_sats"] == split.total_sats
    assert q["loopback"] is False


def test_quote_agrees_with_quote_job(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEISO_PROTOCOL_FEE_BPS", "500")
    listing = _listing(kind="finetune", compute_sats=0, model_or_preset="smoke")
    q = quote_listing(listing)
    job = quote_job("finetune", preset="smoke")
    assert q["compute_sats"] == job["compute_sats"]
    assert q["total_sats"] == job["total_sats"]


def test_loopback_listing_is_free() -> None:
    q = quote_listing(_listing(loopback=True, compute_sats=99_000))
    assert q["price_sats"] == 0
    assert q["total_sats"] == 0
    assert q["compute_sats"] == 0
    assert q["loopback"] is True
    assert q["live_settle_allowed"] is False


@pytest.mark.parametrize(
    "endpoint",
    ["http://127.0.0.1:8787", "http://localhost:1", "http://[::1]/"],
)
def test_loopback_endpoint_is_free(endpoint: str) -> None:
    q = quote_listing(_listing(endpoint=endpoint, compute_sats=5000))
    assert q["price_sats"] == 0
    assert q["loopback"] is True


def test_no_token_field_on_listing_or_quote() -> None:
    q = quote_listing(_listing())
    blob_keys = set(q) | set(q["listing"])
    assert not (blob_keys & FORBIDDEN_LISTING_KEYS)
    assert "seiso_token" not in repr(q).lower()
    assert "airdrop" not in q


def test_rails_include_ark_and_l402() -> None:
    q = quote_listing(_listing())
    ids = {r["id"] for r in q["rails"]}
    assert "ark" in ids
    assert "l402" in ids
    assert "x402" in ids
    ark = next(r for r in q["rails"] if r["id"] == "ark")
    assert ark["status"] == "not_functional"


def test_faucet_rail_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEISO_PAY_FAUCET", "1")
    q = quote_listing(_listing())
    ids = {r["id"] for r in q["rails"]}
    assert "faucet" in ids


def test_live_settle_default_false() -> None:
    assert live_settle_allowed() is False


def test_live_settle_false_with_faucet(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEISO_PAY_FAUCET", "1")
    monkeypatch.setenv("SEISO_PROTOCOL_TREASURY_ARK", "ark1x")
    assert live_settle_allowed(ark_live=True, l402_live=True) is False


def test_live_settle_false_without_treasury(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SEISO_PAY_FAUCET", raising=False)
    monkeypatch.delenv("SEISO_PROTOCOL_TREASURY_ARK", raising=False)
    assert live_settle_allowed(ark_live=True) is False


def test_live_settle_true_only_when_wired(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SEISO_PAY_FAUCET", raising=False)
    monkeypatch.setenv("SEISO_PROTOCOL_TREASURY_ARK", "ark1treasury")
    assert live_settle_allowed(ark_live=True, faucet=False, treasury_set=True) is True
    assert live_settle_allowed(l402_live=True, faucet=False, treasury_set=True) is True
    assert live_settle_allowed(x402_live=True, faucet=False, treasury_set=True) is True
    assert (
        live_settle_allowed(
            ark_live=False,
            l402_live=False,
            x402_live=False,
            faucet=False,
            treasury_set=True,
        )
        is False
    )


def test_inference_token_quote() -> None:
    q = quote_listing(
        _listing(kind="inference", compute_sats=5),
        prompt_tokens=1000,
        completion_tokens=1000,
    )
    assert q["job_type"] == "inference"
    assert q["total_sats"] > 0


def test_listing_as_dict_has_no_coin() -> None:
    data = _listing().as_dict()
    assert "coin" not in data
    assert data["kind"] == "finetune"


def test_quote_hides_l402_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEISO_PAY_L402", "0")
    q = quote_listing(_listing())
    assert all(r["id"] != "l402" for r in q["rails"])
