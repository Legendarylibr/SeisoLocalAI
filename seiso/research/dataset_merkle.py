"""Merkle commitment over training-row fingerprints (corpus membership proofs).

Publishes only the root via Nostr attestation; row text never leaves the machine.
Leaf order is sorted by content fingerprint for canonicality.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

logger = logging.getLogger(__name__)

DATASET_MERKLE_ALG = "seiso.dataset.merkle/v1"
DATASET_MERKLE_SCHEMA = "seiso.dataset.merkle/v1"
MEMBERSHIP_PROOF_SCHEMA = "seiso.dataset.membership_proof/v1"
DEFAULT_MAX_ROWS = 250_000

_LEAF_DOMAIN = b"seiso.dataset.leaf/v1\0"
_NODE_DOMAIN = b"seiso.dataset.node/v1\0"


def dataset_merkle_max_rows() -> int:
    raw = os.environ.get("SEISO_DATASET_MERKLE_MAX_ROWS", "").strip()
    if raw.isdigit():
        return max(1, int(raw))
    return DEFAULT_MAX_ROWS


def dataset_merkle_enabled_by_env() -> bool | None:
    """Return True/False when SEISO_DATASET_MERKLE is set; else None (use config)."""
    raw = os.environ.get("SEISO_DATASET_MERKLE")
    if raw is None or not str(raw).strip():
        return None
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def row_content_fingerprint(row: dict[str, Any]) -> str:
    """Match ``preprocess._content_fingerprint`` (canonical JSON SHA-256 hex)."""
    payload = json.dumps(row, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def fingerprints_from_rows(rows: Iterable[dict[str, Any]]) -> list[str]:
    return sorted({row_content_fingerprint(dict(row)) for row in rows})


def _leaf_hash(run_id: str, fingerprint: str) -> bytes:
    return hashlib.sha256(
        _LEAF_DOMAIN + run_id.encode("utf-8") + b"\0" + fingerprint.encode("utf-8")
    ).digest()


def _node_hash(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(_NODE_DOMAIN + left + right).digest()


def _build_levels(leaves: Sequence[bytes]) -> list[list[bytes]]:
    if not leaves:
        raise ValueError("cannot build merkle tree with zero leaves")
    levels: list[list[bytes]] = [list(leaves)]
    while len(levels[-1]) > 1:
        level = levels[-1]
        nxt: list[bytes] = []
        i = 0
        while i < len(level):
            left = level[i]
            right = level[i + 1] if i + 1 < len(level) else left
            nxt.append(_node_hash(left, right))
            i += 2
        levels.append(nxt)
    return levels


@dataclass(frozen=True)
class MerklePathStep:
    side: str  # "L" = sibling is left of current; "R" = sibling is right
    sibling: str  # hex

    def to_dict(self) -> dict[str, str]:
        return {"side": self.side, "sibling": self.sibling}


@dataclass(frozen=True)
class DatasetMerkleCommit:
    root: str
    leaf_count: int
    run_id: str
    fingerprints: list[str]
    alg: str = DATASET_MERKLE_ALG

    def to_sidecar(self) -> dict[str, Any]:
        return {
            "schema": DATASET_MERKLE_SCHEMA,
            "alg": self.alg,
            "run_id": self.run_id,
            "root": self.root,
            "leaf_count": self.leaf_count,
            "fingerprints": list(self.fingerprints),
        }


def build_dataset_merkle(
    fingerprints: Sequence[str],
    *,
    run_id: str,
) -> DatasetMerkleCommit:
    """Build a merkle commitment over sorted unique fingerprints."""
    rid = (run_id or "").strip()
    if not rid:
        raise ValueError("run_id is required for dataset merkle leaves")
    fps = sorted({str(f).strip().lower() for f in fingerprints if str(f).strip()})
    if not fps:
        raise ValueError("at least one fingerprint is required")
    for fp in fps:
        if len(fp) != 64 or any(c not in "0123456789abcdef" for c in fp):
            raise ValueError(f"invalid fingerprint hex: {fp[:16]}…")
    leaves = [_leaf_hash(rid, fp) for fp in fps]
    levels = _build_levels(leaves)
    root = levels[-1][0].hex()
    return DatasetMerkleCommit(
        root=root, leaf_count=len(fps), run_id=rid, fingerprints=fps
    )


def open_membership_path(
    commit: DatasetMerkleCommit, fingerprint: str
) -> list[MerklePathStep]:
    fp = str(fingerprint).strip().lower()
    try:
        index = commit.fingerprints.index(fp)
    except ValueError as exc:
        raise KeyError(f"fingerprint not in committed corpus: {fp[:16]}…") from exc
    leaves = [_leaf_hash(commit.run_id, f) for f in commit.fingerprints]
    levels = _build_levels(leaves)
    path: list[MerklePathStep] = []
    idx = index
    for level in levels[:-1]:
        if idx % 2 == 0:
            sibling_idx = idx + 1 if idx + 1 < len(level) else idx
            side = "R"
        else:
            sibling_idx = idx - 1
            side = "L"
        path.append(
            MerklePathStep(side=side, sibling=level[sibling_idx].hex())
        )
        idx //= 2
    return path


def verify_membership_path(
    *,
    run_id: str,
    fingerprint: str,
    root: str,
    path: Sequence[dict[str, str] | MerklePathStep],
) -> bool:
    fp = str(fingerprint).strip().lower()
    current = _leaf_hash(run_id, fp)
    for step in path:
        if isinstance(step, MerklePathStep):
            side, sibling_hex = step.side, step.sibling
        else:
            side = str(step.get("side") or "")
            sibling_hex = str(step.get("sibling") or "")
        try:
            sibling = bytes.fromhex(sibling_hex)
        except ValueError:
            return False
        if len(sibling) != 32:
            return False
        if side == "L":
            current = _node_hash(sibling, current)
        elif side == "R":
            current = _node_hash(current, sibling)
        else:
            return False
    return current.hex() == str(root).strip().lower()


def build_membership_proof(
    commit: DatasetMerkleCommit, fingerprint: str
) -> dict[str, Any]:
    path = open_membership_path(commit, fingerprint)
    return {
        "schema": MEMBERSHIP_PROOF_SCHEMA,
        "run_id": commit.run_id,
        "dataset_merkle_root": commit.root,
        "dataset_merkle_alg": commit.alg,
        "fingerprint": str(fingerprint).strip().lower(),
        "index": commit.fingerprints.index(str(fingerprint).strip().lower()),
        "path": [p.to_dict() for p in path],
    }


def verify_membership_proof(proof: dict[str, Any], *, root: str | None = None) -> bool:
    if not isinstance(proof, dict):
        return False
    expected_root = str(root or proof.get("dataset_merkle_root") or "").strip().lower()
    run_id = str(proof.get("run_id") or "").strip()
    fingerprint = str(proof.get("fingerprint") or "").strip().lower()
    path = proof.get("path")
    if not expected_root or not run_id or not fingerprint or not isinstance(path, list):
        return False
    return verify_membership_path(
        run_id=run_id,
        fingerprint=fingerprint,
        root=expected_root,
        path=path,
    )


def write_dataset_merkle_sidecar(path: Path, commit: DatasetMerkleCommit) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(commit.to_sidecar(), indent=2), encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        logger.debug("chmod 0600 skipped for %s", path)
    return path


def load_dataset_merkle_sidecar(path: Path) -> DatasetMerkleCommit:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("dataset_merkle sidecar must be a JSON object")
    fps = data.get("fingerprints")
    if not isinstance(fps, list) or not fps:
        raise ValueError("dataset_merkle sidecar missing fingerprints")
    return DatasetMerkleCommit(
        root=str(data["root"]),
        leaf_count=int(data.get("leaf_count") or len(fps)),
        run_id=str(data["run_id"]),
        fingerprints=[str(f).strip().lower() for f in fps],
        alg=str(data.get("alg") or DATASET_MERKLE_ALG),
    )


def collect_fingerprints_from_hf_dataset(dataset: Any, *, max_rows: int) -> list[str]:
    """Fingerprint rows from a HuggingFace-like dataset (``__len__`` / indexable)."""
    n = len(dataset)
    if n > max_rows:
        raise ValueError(
            f"dataset has {n} rows; exceeds merkle cap {max_rows} "
            "(raise SEISO_DATASET_MERKLE_MAX_ROWS or disable dataset_merkle)"
        )
    fps: set[str] = set()
    for i in range(n):
        row = dataset[i]
        if not isinstance(row, dict):
            row = dict(row)
        # Drop internal preprocess columns if still present.
        cleaned = {
            k: v
            for k, v in row.items()
            if not str(k).startswith("_seiso")
        }
        fps.add(row_content_fingerprint(cleaned))
    return sorted(fps)
