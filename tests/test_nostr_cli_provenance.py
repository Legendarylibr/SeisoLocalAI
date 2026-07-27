"""CLI coverage for `seiso provenance` (mocked relays)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

from typer.testing import CliRunner

from seiso.research.dataset_merkle import (
    build_dataset_merkle,
    build_membership_proof,
    row_content_fingerprint,
    write_dataset_merkle_sidecar,
)
from seiso_cli.main import app

runner = CliRunner()


def test_provenance_keygen_show_attest_verify(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("SEISO_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SEISO_ALLOW_NOSTR", "1")

    result = runner.invoke(app, ["provenance", "keygen", "--identity", "cli"])
    assert result.exit_code == 0, result.output
    assert "npub:" in result.output

    manifest = {
        "pipeline": "compress",
        "run_id": "cli-run",
        "config_fingerprint": "ab" * 32,
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    show = runner.invoke(app, ["provenance", "show", str(path)])
    assert show.exit_code == 0, show.output
    assert "attestation" in show.output

    store: dict[str, Any] = {}

    def fake_publish(event, relays, **kwargs):
        store["event"] = event
        return list(relays)

    def fake_fetch(event_id, relays, **kwargs):
        ev = store.get("event")
        return ev if ev and ev.get("id") == event_id else None

    with (
        patch("seiso.research.nostr.attest.publish_event", side_effect=fake_publish),
        patch("seiso.research.nostr.attest.fetch_event_by_id", side_effect=fake_fetch),
        patch(
            "seiso.research.nostr.attest.normalize_relay_list",
            side_effect=lambda relays, **kw: list(relays),
        ),
    ):
        attest = runner.invoke(
            app,
            [
                "provenance",
                "attest",
                str(path),
                "--relay",
                "wss://relay.example.com",
            ],
        )
        assert attest.exit_code == 0, attest.output
        verify = runner.invoke(
            app,
            [
                "provenance",
                "verify",
                str(path),
                "--relay",
                "wss://relay.example.com",
            ],
        )
        assert verify.exit_code == 0, verify.output
        local = runner.invoke(
            app, ["provenance", "verify", str(path), "--local-only"]
        )
        assert local.exit_code == 0, local.output


def test_dataset_prove_and_verify_local(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("SEISO_DATA_DIR", str(tmp_path))
    rows = [{"text": f"r{i}"} for i in range(4)]
    fps = [row_content_fingerprint(r) for r in rows]
    commit = build_dataset_merkle(fps, run_id="ds-1")
    write_dataset_merkle_sidecar(tmp_path / "dataset_merkle.json", commit)
    man = {
        "method": "lora",
        "run_id": "ds-1",
        "dataset_merkle_root": commit.root,
        "dataset_merkle_leaf_count": commit.leaf_count,
        "dataset_merkle_alg": commit.alg,
        "dataset_merkle_sidecar": "dataset_merkle.json",
    }
    man_path = tmp_path / "seiso_manifest.json"
    man_path.write_text(json.dumps(man), encoding="utf-8")
    row_path = tmp_path / "row.json"
    row_path.write_text(json.dumps(rows[1]), encoding="utf-8")
    proof_path = tmp_path / "proof.json"

    prove = runner.invoke(
        app,
        [
            "provenance",
            "dataset-prove",
            str(man_path),
            "--row",
            str(row_path),
            "-o",
            str(proof_path),
        ],
    )
    assert prove.exit_code == 0, prove.output
    assert proof_path.is_file()

    verify = runner.invoke(
        app,
        [
            "provenance",
            "dataset-verify-proof",
            str(proof_path),
            "--local-only",
        ],
    )
    assert verify.exit_code == 0, verify.output

    # Membership proof helper consistency.
    proof = build_membership_proof(commit, fps[1])
    assert proof["dataset_merkle_root"] == commit.root
