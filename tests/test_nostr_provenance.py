"""Nostr provenance attestation unit tests (mocked relays, no network)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from seiso.research.nostr.bech32 import bech32_decode, bech32_encode
from seiso.research.nostr.events import (
    SEISO_PROVENANCE_KIND,
    build_attestation_event,
    verify_event,
)
from seiso.research.nostr.keys import generate_keypair, keypair_from_secret, save_keypair
from seiso.research.nostr.policy import validate_relay_url
from seiso.research.nostr.schnorr import (
    pubkey_xonly_from_secret,
    sign_schnorr,
    verify_schnorr,
)
from seiso.research.provenance import (
    ATTESTATION_SCHEMA_V1,
    build_attestation_v1,
    content_fingerprint,
    manifest_sha256_excluding_nostr,
    merge_nostr_receipt,
)
from seiso.security import SecurityError


def test_bip340_pubkey_vector():
    pk = pubkey_xonly_from_secret(bytes.fromhex("00" * 31 + "01"))
    assert pk.hex().upper() == (
        "79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798"
    )


def test_bip340_official_vector_0():
    """BIP-340 CSV vector index 0 (32-byte message)."""
    secret = bytes.fromhex("00" * 31 + "03")
    pk = pubkey_xonly_from_secret(secret)
    assert pk.hex().upper() == (
        "F9308A019258C31049344F85F89D5229B531C845836F99B08601F113BCE036F9"
    )
    msg = bytes.fromhex("00" * 32)
    aux = bytes.fromhex("00" * 32)
    sig = sign_schnorr(secret, msg, aux_rand=aux)
    expected = bytes.fromhex(
        "E907831F80848D1069A5371B402410364BDF1C5F8307B0084C55F1CE2DCA8215"
        "25F66A4A85EA8B71E482A74F382D2CE5EBEEE8FDB2172F477DF4900D310536C0"
    )
    assert sig == expected
    assert verify_schnorr(pk, msg, expected)


def test_schnorr_sign_verify_roundtrip():
    secret = bytes.fromhex("00" * 31 + "01")
    pk = pubkey_xonly_from_secret(secret)
    msg = bytes.fromhex("11" * 32)
    sig = sign_schnorr(secret, msg, aux_rand=bytes.fromhex("22" * 32))
    assert verify_schnorr(pk, msg, sig)
    assert not verify_schnorr(pk, bytes.fromhex("33" * 32), sig)


def test_bech32_nsec_npub_roundtrip():
    pair = generate_keypair()
    hrp, data = bech32_decode(pair.nsec)
    assert hrp == "nsec"
    assert data.hex() == pair.secret_hex
    assert bech32_encode("npub", bytes.fromhex(pair.public_hex)) == pair.npub
    restored = keypair_from_secret(pair.nsec)
    assert restored.public_hex == pair.public_hex


def test_event_sign_verify():
    pair = generate_keypair()
    event = build_attestation_event(
        pair=pair,
        attestation_json='{"schema":"seiso.provenance.attestation/v1","pipeline":"compress"}',
        pipeline="compress",
        run_id="abc",
        created_at=1_700_000_000,
    )
    assert event["kind"] == SEISO_PROVENANCE_KIND
    assert ["d", "compress:abc"] in event["tags"]
    assert verify_event(event)
    bad = dict(event)
    bad["content"] = '{"tampered":true}'
    assert not verify_event(bad)


def test_infer_run_id_hashes_directory_fallback(tmp_path: Path):
    from seiso.research.provenance import infer_pipeline_and_run_id

    man = tmp_path / "SecretProjectName" / "manifest.json"
    man.parent.mkdir(parents=True)
    man.write_text("{}", encoding="utf-8")
    pipeline, run_id = infer_pipeline_and_run_id({}, manifest_path=man)
    assert pipeline == "seiso"
    assert run_id != "SecretProjectName"
    assert len(run_id) == 16
    assert run_id.isalnum()


def test_attestation_excludes_nostr_receipt():
    manifest = {
        "pipeline": "compress",
        "run_id": "r1",
        "config_fingerprint": "aa" * 32,
        "created_at": "2026-01-01T00:00:00+00:00",
        "git_commit": "deadbeef",
        "stages": [{"name": "distill", "sha256": "bb" * 32}],
        "nostr": {
            "event_id": "cc" * 32,
            "attestation_sha256": "should-not-affect",
        },
    }
    a1 = build_attestation_v1(manifest)
    stripped = {k: v for k, v in manifest.items() if k != "nostr"}
    a2 = build_attestation_v1(stripped)
    assert a1["schema"] == ATTESTATION_SCHEMA_V1
    assert a1["manifest_sha256"] == a2["manifest_sha256"]
    assert a1["manifest_sha256"] == manifest_sha256_excluding_nostr(manifest)
    # Receipt mutation must not change attestation hash of body.
    mutated = merge_nostr_receipt(manifest, {"event_id": "dd" * 32})
    assert build_attestation_v1(mutated)["manifest_sha256"] == a1["manifest_sha256"]


def test_validate_relay_url_blocks_private(monkeypatch):
    with pytest.raises(SecurityError, match="not allowed|blocked"):
        validate_relay_url("wss://127.0.0.1:8080")
    with pytest.raises(SecurityError, match="not allowed|blocked|private"):
        validate_relay_url("wss://192.168.1.10")
    with pytest.raises(SecurityError, match="scheme"):
        validate_relay_url("https://relay.example.com")
    loop = validate_relay_url("ws://127.0.0.1:8080", allow_loopback=True)
    assert loop.startswith("ws://127.0.0.1")
    with pytest.raises(SecurityError, match="allowlist"):
        validate_relay_url(
            "wss://relay.example.com",
            allowlist=["other.example.com"],
        )


def test_attest_and_verify_with_mocked_relay(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("SEISO_ALLOW_NOSTR", "1")
    monkeypatch.setenv("SEISO_DATA_DIR", str(tmp_path))
    pair = generate_keypair()
    save_keypair(pair, identity="cli", data_dir=tmp_path)

    manifest = {
        "pipeline": "compress",
        "run_id": "run42",
        "config_fingerprint": "ab" * 32,
        "created_at": "2026-01-01T00:00:00+00:00",
        "stages": [{"name": "prune", "sha256": "cd" * 32}],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    store: dict[str, Any] = {}

    def fake_publish(event, relays, **kwargs):
        store["event"] = event
        return list(relays)

    def fake_fetch(event_id, relays, **kwargs):
        ev = store.get("event")
        if ev and ev.get("id") == event_id:
            return ev
        return None

    from seiso.research.nostr import attest as attest_mod

    with (
        patch.object(attest_mod, "publish_event", side_effect=fake_publish),
        patch.object(attest_mod, "fetch_event_by_id", side_effect=fake_fetch),
        patch.object(
            attest_mod,
            "normalize_relay_list",
            side_effect=lambda relays, **kw: list(relays),
        ),
    ):
        report = attest_mod.attest_manifest(
            path,
            relays=["wss://relay.example.com"],
            identity="cli",
            data_dir=tmp_path,
            require_allow=True,
        )
        assert report["ok"]
        assert report["event_id"]
        saved = json.loads(path.read_text(encoding="utf-8"))
        assert saved["nostr"]["event_id"] == report["event_id"]

        verify = attest_mod.verify_attestation(
            path,
            relays=["wss://relay.example.com"],
            require_network=True,
        )
        assert verify["ok"]
        assert verify["event_verified"] is True

        # Tamper local digests → verify fails.
        saved["config_fingerprint"] = "ff" * 32
        path.write_text(json.dumps(saved, indent=2), encoding="utf-8")
        bad = attest_mod.verify_attestation(
            path,
            relays=["wss://relay.example.com"],
            require_network=True,
        )
        assert bad["ok"] is False


def test_maybe_auto_attest_respects_env(tmp_path: Path, monkeypatch):
    from seiso.research.nostr.attest import maybe_auto_attest

    # Gate on by default; auto-attest still requires SEISO_NOSTR_ATTEST.
    monkeypatch.delenv("SEISO_ALLOW_NOSTR", raising=False)
    monkeypatch.delenv("SEISO_NOSTR_ATTEST", raising=False)
    assert maybe_auto_attest(tmp_path / "missing.json") is None

    monkeypatch.setenv("SEISO_ALLOW_NOSTR", "0")
    monkeypatch.setenv("SEISO_NOSTR_ATTEST", "1")
    assert maybe_auto_attest(tmp_path / "missing.json") is None


def test_nostr_allowed_default_on(monkeypatch):
    from seiso.research.nostr.policy import nostr_allowed, relay_allowlist_from_env

    monkeypatch.delenv("SEISO_ALLOW_NOSTR", raising=False)
    monkeypatch.delenv("SEISO_NOSTR_RELAYS", raising=False)
    assert nostr_allowed() is True
    assert relay_allowlist_from_env() == [
        "wss://nos.lol",
        "wss://relay.damus.io",
    ]

    monkeypatch.setenv("SEISO_ALLOW_NOSTR", "0")
    assert nostr_allowed() is False
    assert relay_allowlist_from_env() == []


def test_content_fingerprint_stable_for_attestation_body():
    body = {"a": 1, "b": [2, 3]}
    assert content_fingerprint(body) == content_fingerprint({"b": [2, 3], "a": 1})


def test_bip340_vector_odd_y_secret():
    """Secret whose P has odd Y must still verify (BIP-340 d negation)."""
    # Secret 2 → G*2; exercise sign/verify path regardless of Y parity.
    secret = bytes.fromhex("00" * 31 + "02")
    pk = pubkey_xonly_from_secret(secret)
    msg = bytes.fromhex("aa" * 32)
    sig = sign_schnorr(secret, msg, aux_rand=bytes.fromhex("bb" * 32))
    assert verify_schnorr(pk, msg, sig)
    # Truncated / out-of-range inputs rejected.
    assert not verify_schnorr(pk[:31], msg, sig)
    assert not verify_schnorr(pk, msg, sig[:63])


def test_event_id_is_nip01_sha256():
    from seiso.research.nostr.events import compute_event_id

    event_id = compute_event_id(
        pubkey="aa" * 32,
        created_at=123,
        kind=1,
        tags=[["t", "x"]],
        content="hi",
    )
    import hashlib
    import json

    preimage = json.dumps(
        [0, "aa" * 32, 123, 1, [["t", "x"]], "hi"],
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    assert event_id == hashlib.sha256(preimage).hexdigest()


def test_validate_relay_url_rejects_embedded_credentials_and_metadata():
    with pytest.raises(SecurityError, match="credentials"):
        validate_relay_url("wss://user:secret@nos.lol")
    with pytest.raises(SecurityError, match="not allowed"):
        validate_relay_url("wss://metadata.google.internal")
