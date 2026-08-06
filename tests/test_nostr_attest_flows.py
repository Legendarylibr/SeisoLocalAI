"""End-to-end Nostr attest/verify flows (mocked relays, no network)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from seiso.research.dataset_merkle import DATASET_MERKLE_ALG
from seiso.research.nostr.events import (
    SEISO_PROVENANCE_KIND,
    build_attestation_event,
    compute_event_id,
    verify_event,
)
from seiso.research.nostr.keys import generate_keypair, save_keypair
from seiso.research.provenance import (
    ATTESTATION_SCHEMA_V2,
    attestation_content_json,
    build_attestation_v1,
)
from seiso.security import SecurityError


def _write_manifest(path: Path, **fields: Any) -> Path:
    body = {
        "pipeline": "compress",
        "run_id": "run42",
        "config_fingerprint": "ab" * 32,
        "created_at": "2026-01-01T00:00:00+00:00",
        "stages": [{"name": "prune", "sha256": "cd" * 32}],
    }
    body.update(fields)
    path.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return path


def _patch_relays(attest_mod, store: dict[str, Any]):
    def fake_publish(event, relays, **kwargs):
        store["event"] = event
        store["relays"] = list(relays)
        return list(relays)

    def fake_fetch(event_id, relays, **kwargs):
        ev = store.get("event")
        if ev and str(ev.get("id") or "").lower() == str(event_id).lower():
            return ev
        return store.get("fetch_override")

    def fake_fetch_addressable(*, pubkey, kind, d_tag, relays, **kwargs):
        if store.get("addressable_disabled"):
            return None
        ev = store.get("addressable_event") or store.get("event")
        if not ev:
            return None
        if str(ev.get("pubkey") or "").lower() != str(pubkey).lower():
            return None
        if int(ev.get("kind")) != int(kind):
            return None
        d_tags = [
            str(t[1])
            for t in (ev.get("tags") or [])
            if isinstance(t, list) and len(t) >= 2 and str(t[0]) == "d"
        ]
        if d_tag not in d_tags:
            return None
        return ev

    return (
        patch.object(attest_mod, "publish_event", side_effect=fake_publish),
        patch.object(attest_mod, "fetch_event_by_id", side_effect=fake_fetch),
        patch.object(attest_mod, "fetch_addressable_event", side_effect=fake_fetch_addressable),
        patch.object(
            attest_mod,
            "normalize_relay_list",
            side_effect=lambda relays, **kw: list(relays),
        ),
    )


def test_attest_disabled_and_missing_key(tmp_path: Path, monkeypatch):
    from seiso.research.nostr import attest as attest_mod

    monkeypatch.setenv("SEISO_ALLOW_NOSTR", "0")
    path = _write_manifest(tmp_path / "m.json")
    with pytest.raises(SecurityError, match="disabled"):
        attest_mod.attest_manifest(path, relays=["wss://relay.example.com"])

    monkeypatch.setenv("SEISO_ALLOW_NOSTR", "1")
    with pytest.raises(SecurityError, match="No Nostr key"):
        attest_mod.attest_manifest(
            path,
            relays=["wss://relay.example.com"],
            identity="cli",
            data_dir=tmp_path,
        )


def test_attest_v2_merkle_and_verify_local_only(tmp_path: Path, monkeypatch):
    from seiso.research.nostr import attest as attest_mod

    monkeypatch.setenv("SEISO_ALLOW_NOSTR", "1")
    pair = generate_keypair()
    save_keypair(pair, identity="cli", data_dir=tmp_path)
    path = tmp_path / "seiso_manifest.json"
    path.write_text(
        json.dumps(
            {
                "method": "lora",
                "run_id": "train-1",
                "config_fingerprint": "ab" * 32,
                "created_at": "2026-01-01T00:00:00+00:00",
                "dataset_merkle_root": "ee" * 32,
                "dataset_merkle_leaf_count": 3,
                "dataset_merkle_alg": DATASET_MERKLE_ALG,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    store: dict[str, Any] = {}
    patches = _patch_relays(attest_mod, store)
    with patches[0], patches[1], patches[2], patches[3]:
        report = attest_mod.attest_manifest(
            path,
            relays=["wss://relay.example.com"],
            identity="cli",
            data_dir=tmp_path,
        )
        assert report["ok"]
        att = report["attestation"]
        assert att["schema"] == ATTESTATION_SCHEMA_V2
        assert att["pipeline"] == "training"
        assert att["dataset_merkle_root"] == "ee" * 32
        assert store["event"]["kind"] == SEISO_PROVENANCE_KIND
        assert ["d", "training:train-1"] in store["event"]["tags"]

        local = attest_mod.verify_attestation(path, require_network=False)
        assert local["local_attestation_match"] is True
        assert local["ok"] is True
        assert local["event_verified"] is None


def test_verify_rejects_wrong_pubkey_kind_or_d_tag(tmp_path: Path, monkeypatch):
    from seiso.research.nostr import attest as attest_mod
    from seiso.research.nostr.schnorr import sign_schnorr

    monkeypatch.setenv("SEISO_ALLOW_NOSTR", "1")
    pair = generate_keypair()
    save_keypair(pair, identity="cli", data_dir=tmp_path)
    path = _write_manifest(tmp_path / "m.json")
    store: dict[str, Any] = {}
    patches = _patch_relays(attest_mod, store)
    with patches[0], patches[1], patches[2], patches[3]:
        attest_mod.attest_manifest(
            path,
            relays=["wss://relay.example.com"],
            identity="cli",
            data_dir=tmp_path,
        )
        original = dict(store["event"])

        # Receipt pubkey tampered while event_id still points at the real event.
        saved = json.loads(path.read_text(encoding="utf-8"))
        other = generate_keypair()
        saved["nostr"]["pubkey"] = other.public_hex
        path.write_text(json.dumps(saved, indent=2), encoding="utf-8")
        store["event"] = original
        bad_pk = attest_mod.verify_attestation(
            path, relays=["wss://relay.example.com"], require_network=True
        )
        assert bad_pk["ok"] is False
        assert bad_pk["event_pubkey_match"] is False

        # Restore receipt pubkey; point receipt at a validly signed wrong-kind event.
        saved["nostr"]["pubkey"] = original["pubkey"]
        wrong_kind = dict(original)
        wrong_kind["kind"] = 1
        wrong_kind["id"] = compute_event_id(
            pubkey=wrong_kind["pubkey"],
            created_at=wrong_kind["created_at"],
            kind=1,
            tags=wrong_kind["tags"],
            content=wrong_kind["content"],
        )
        wrong_kind["sig"] = sign_schnorr(
            bytes.fromhex(pair.secret_hex), bytes.fromhex(wrong_kind["id"])
        ).hex()
        assert verify_event(wrong_kind)
        saved["nostr"]["event_id"] = wrong_kind["id"]
        path.write_text(json.dumps(saved, indent=2), encoding="utf-8")
        store["event"] = wrong_kind
        bad_kind = attest_mod.verify_attestation(
            path, relays=["wss://relay.example.com"], require_network=True
        )
        assert bad_kind["ok"] is False
        assert bad_kind["event_kind_ok"] is False

        # Wrong d-tag (keep provenance kind).
        wrong_d = dict(original)
        wrong_d["tags"] = [
            ["d", "other:run"],
            ["t", "seiso-provenance"],
            ["client", "seiso"],
        ]
        wrong_d["id"] = compute_event_id(
            pubkey=wrong_d["pubkey"],
            created_at=wrong_d["created_at"],
            kind=wrong_d["kind"],
            tags=wrong_d["tags"],
            content=wrong_d["content"],
        )
        wrong_d["sig"] = sign_schnorr(
            bytes.fromhex(pair.secret_hex), bytes.fromhex(wrong_d["id"])
        ).hex()
        saved["nostr"]["event_id"] = wrong_d["id"]
        path.write_text(json.dumps(saved, indent=2), encoding="utf-8")
        store["event"] = wrong_d
        bad_d = attest_mod.verify_attestation(
            path, relays=["wss://relay.example.com"], require_network=True
        )
        assert bad_d["ok"] is False
        assert bad_d["event_d_tag_ok"] is False


def test_verify_missing_receipt_and_event_not_found(tmp_path: Path, monkeypatch):
    from seiso.research.nostr import attest as attest_mod

    monkeypatch.setenv("SEISO_ALLOW_NOSTR", "1")
    path = _write_manifest(tmp_path / "m.json")
    missing = attest_mod.verify_attestation(path, require_network=False)
    assert missing["ok"] is False
    assert "no nostr receipt" in missing["error"]

    pair = generate_keypair()
    save_keypair(pair, identity="cli", data_dir=tmp_path)
    store: dict[str, Any] = {}
    patches = _patch_relays(attest_mod, store)
    with patches[0], patches[1], patches[2], patches[3]:
        attest_mod.attest_manifest(
            path,
            relays=["wss://relay.example.com"],
            identity="cli",
            data_dir=tmp_path,
        )
        store.pop("event", None)
        store["addressable_disabled"] = True
        not_found = attest_mod.verify_attestation(
            path, relays=["wss://relay.example.com"], require_network=True
        )
        assert not_found["ok"] is False
        assert "not found" in not_found["error"]


def test_verify_network_blocked_when_nostr_disabled(tmp_path: Path, monkeypatch):
    from seiso.research.nostr import attest as attest_mod

    monkeypatch.setenv("SEISO_ALLOW_NOSTR", "1")
    pair = generate_keypair()
    save_keypair(pair, identity="cli", data_dir=tmp_path)
    path = _write_manifest(tmp_path / "m.json")
    store: dict[str, Any] = {}
    patches = _patch_relays(attest_mod, store)
    with patches[0], patches[1], patches[2], patches[3]:
        attest_mod.attest_manifest(
            path,
            relays=["wss://relay.example.com"],
            identity="cli",
            data_dir=tmp_path,
        )
    monkeypatch.setenv("SEISO_ALLOW_NOSTR", "0")
    report = attest_mod.verify_attestation(
        path, relays=["wss://relay.example.com"], require_network=True
    )
    assert report["ok"] is False
    assert "disabled" in report["error"]


def test_maybe_auto_attest_success_and_nonfatal_failure(tmp_path: Path, monkeypatch):
    from seiso.research.nostr import attest as attest_mod

    monkeypatch.setenv("SEISO_ALLOW_NOSTR", "1")
    monkeypatch.setenv("SEISO_NOSTR_ATTEST", "1")
    pair = generate_keypair()
    save_keypair(pair, identity="cli", data_dir=tmp_path)
    path = _write_manifest(tmp_path / "m.json")
    store: dict[str, Any] = {}
    patches = _patch_relays(attest_mod, store)
    with patches[0], patches[1], patches[2], patches[3]:
        ok = attest_mod.maybe_auto_attest(
            path, identity="cli", data_dir=tmp_path, relays=["wss://relay.example.com"]
        )
        assert ok is not None and ok["ok"] is True

        # Non-fatal: publish raises → warning dict, no exception.
        with patch.object(attest_mod, "attest_manifest", side_effect=RuntimeError("relay down")):
            failed = attest_mod.maybe_auto_attest(
                path,
                identity="cli",
                data_dir=tmp_path,
                relays=["wss://relay.example.com"],
            )
        assert failed is not None
        assert failed["ok"] is False
        assert "relay down" in failed["error"]


def test_verify_event_rejects_malformed_and_bad_id():
    pair = generate_keypair()
    event = build_attestation_event(
        pair=pair,
        attestation_json='{"schema":"seiso.provenance.attestation/v1","pipeline":"x","run_id":"y"}',
        pipeline="x",
        run_id="y",
        created_at=100,
    )
    assert verify_event(event)
    bad_id = dict(event)
    bad_id["id"] = "00" * 32
    assert not verify_event(bad_id)
    assert not verify_event({"pubkey": "nope"})
    assert not verify_event({})


def test_verify_falls_back_to_addressable_filter(tmp_path: Path, monkeypatch):
    """When relays replace an addressable event, verify by (pubkey, kind, #d)."""
    from seiso.research.nostr import attest as attest_mod

    monkeypatch.setenv("SEISO_ALLOW_NOSTR", "1")
    pair = generate_keypair()
    save_keypair(pair, identity="cli", data_dir=tmp_path)
    path = _write_manifest(tmp_path / "m.json")
    store: dict[str, Any] = {}
    patches = _patch_relays(attest_mod, store)
    with patches[0], patches[1], patches[2], patches[3]:
        attest_mod.attest_manifest(
            path,
            relays=["wss://relay.example.com"],
            identity="cli",
            data_dir=tmp_path,
        )
        # Simulate relay discarding the pinned id but keeping the addressable slot.
        pinned = dict(store["event"])
        store["addressable_event"] = pinned
        store.pop("event", None)
        saved = json.loads(path.read_text(encoding="utf-8"))
        saved["nostr"]["event_id"] = "00" * 32  # stale / replaced id
        path.write_text(json.dumps(saved, indent=2), encoding="utf-8")

        report = attest_mod.verify_attestation(
            path, relays=["wss://relay.example.com"], require_network=True
        )
        assert report["ok"] is True
        assert report["event_fetch"] == "addressable"
        assert report["event_verified"] is True


def test_infer_pipeline_variants_and_stable_content():
    export_m = {
        "format": "gguf",
        "file_checksums_sha256": {"a.gguf": "aa" * 32},
        "exported_at": "2026-01-01T00:00:00+00:00",
    }
    att = build_attestation_v1(export_m, manifest_path=Path("/tmp/out/meta.json"))
    assert att["pipeline"] == "export"
    # Directory-name fallback is hashed so d-tags do not leak path labels.
    assert att["run_id"] != "out"
    assert len(att["run_id"]) == 16
    assert att["run_id"].isalnum()

    body = build_attestation_v1(
        {
            "pipeline": "compress",
            "run_id": "r",
            "config_fingerprint": "aa" * 32,
        }
    )
    assert json.loads(attestation_content_json(body)) == body
