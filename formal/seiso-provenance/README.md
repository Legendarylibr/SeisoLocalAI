# SeisoProvenance (Lean 4)

Formal model of Seiso’s Nostr-sealed dataset membership proof:

- Merkle tree from `seiso/research/dataset_merkle.py` (odd-length node duplication,
  `L`/`R` path steps, path fold = verify)
- Nostr attestation binding (kind `31250`, digests-only v2 root seal)

## Theorems

| Theorem | Meaning |
|---------|---------|
| `Merkle.openPath_reaches_root` | Honest path opening always verifies against the corpus root (completeness) |
| `Nostr.nostr_membership_complete` | If a valid event seals that root, the opened membership proof satisfies the Nostr-backed verifier |
| `Nostr.nostr_membership_implies_path_to_sealed_root` | A successful check implies event validity + path folds to the sealed root |

SHA-256 / BIP-340 are abstracted (`opaque nodeHash`, `opaque validSchnorr`). Completeness
does not rely on collision resistance; computational soundness against forged paths to a
*fixed* sealed root is the usual Merkle + CRHF argument and is not re-proved here.

## Build

Requires [elan](https://github.com/leanprover/elan) (Lean 4.16 pinned in `lean-toolchain`):

```bash
cd formal/seiso-provenance
lake build
```

## Mapping to Python

| Lean | Python |
|------|--------|
| `nextLevel` | `_build_levels` pairing loop |
| `openPath` | `open_membership_path` |
| `foldPath` / `pathReaches` | `verify_membership_path` |
| `AttestationV2.datasetMerkleRoot` | attestation v2 `dataset_merkle_root` |
| `nostrMembershipValid` | `dataset-verify-proof` (path + sealed root + event checks) |
