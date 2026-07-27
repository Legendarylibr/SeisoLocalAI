"""Dataset merkle membership proofs (no network)."""

from __future__ import annotations

from pathlib import Path

import pytest

from seiso.research.dataset_merkle import (
    DATASET_MERKLE_ALG,
    MEMBERSHIP_PROOF_SCHEMA,
    build_dataset_merkle,
    build_membership_proof,
    load_dataset_merkle_sidecar,
    row_content_fingerprint,
    verify_membership_proof,
    write_dataset_merkle_sidecar,
)
from seiso.research.provenance import (
    ATTESTATION_SCHEMA_V1,
    ATTESTATION_SCHEMA_V2,
    build_attestation_v1,
)
from seiso.training.preprocess import _content_fingerprint


def test_row_fingerprint_matches_preprocess():
    row = {"text": "hello", "meta": 1}
    assert row_content_fingerprint(row) == _content_fingerprint(row)


def test_merkle_inclusion_and_exclusion():
    rows = [{"text": f"row-{i}"} for i in range(5)]
    fps = [row_content_fingerprint(r) for r in rows]
    commit = build_dataset_merkle(fps, run_id="run-a")
    assert commit.leaf_count == 5
    assert commit.alg == DATASET_MERKLE_ALG

    proof = build_membership_proof(commit, fps[2])
    assert proof["schema"] == MEMBERSHIP_PROOF_SCHEMA
    assert verify_membership_proof(proof) is True

    # Wrong root
    bad = dict(proof)
    bad["dataset_merkle_root"] = "00" * 32
    assert verify_membership_proof(bad) is False

    # Row not in corpus
    outsider = row_content_fingerprint({"text": "not-in-set"})
    with pytest.raises(KeyError):
        build_membership_proof(commit, outsider)


def test_merkle_tamper_fingerprint_fails():
    fps = [row_content_fingerprint({"text": x}) for x in ("a", "b", "c")]
    commit = build_dataset_merkle(fps, run_id="r1")
    proof = build_membership_proof(commit, fps[0])
    proof["fingerprint"] = fps[1]
    assert verify_membership_proof(proof) is False


def test_odd_leaf_count_and_sidecar(tmp_path: Path):
    fps = [row_content_fingerprint({"i": i}) for i in range(3)]
    commit = build_dataset_merkle(fps, run_id="odd")
    path = write_dataset_merkle_sidecar(tmp_path / "dataset_merkle.json", commit)
    loaded = load_dataset_merkle_sidecar(path)
    assert loaded.root == commit.root
    assert verify_membership_proof(build_membership_proof(loaded, fps[1]))


def test_attestation_v2_includes_merkle_fields():
    manifest = {
        "method": "lora",
        "run_id": "ckpt-1",
        "config_fingerprint": "aa" * 32,
        "created_at": "2026-01-01T00:00:00+00:00",
        "dataset_merkle_root": "bb" * 32,
        "dataset_merkle_leaf_count": 10,
        "dataset_merkle_alg": DATASET_MERKLE_ALG,
    }
    att = build_attestation_v1(manifest)
    assert att["schema"] == ATTESTATION_SCHEMA_V2
    assert att["dataset_merkle_root"] == "bb" * 32
    assert att["dataset_merkle_leaf_count"] == 10
    assert att["dataset_merkle_alg"] == DATASET_MERKLE_ALG

    plain = {
        "pipeline": "compress",
        "run_id": "r1",
        "config_fingerprint": "aa" * 32,
    }
    att1 = build_attestation_v1(plain)
    assert att1["schema"] == ATTESTATION_SCHEMA_V1
    assert "dataset_merkle_root" not in att1


def test_canonical_order_independent_of_input_order():
    fps = [row_content_fingerprint({"k": k}) for k in ("z", "a", "m")]
    a = build_dataset_merkle(fps, run_id="x")
    b = build_dataset_merkle(list(reversed(fps)), run_id="x")
    assert a.root == b.root
    assert a.fingerprints == b.fingerprints
