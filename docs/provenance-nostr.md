# Nostr provenance attestation

Seiso can publish a small signed commitment of a run’s local manifest digests
to Nostr relays (default path on; digests only). Weights and full manifests
stay on disk (or Hugging Face). This does **not** resume training and is
**not** a social client.

## What it proves

After a compress / distill-rl / adaptive_quant / training / export run writes a
manifest, Seiso can seal:

- `manifest_sha256` (local JSON with the mutable `nostr` receipt excluded)
- `config_fingerprint` / chain digests when present
- `git_commit`, `run_id`, pipeline id, Seiso version
- For training (when enabled): `dataset_merkle_root` + leaf count (see below)

The event is kind **`31250`** (parameterized replaceable) with tags
`d=<pipeline>:<run_id>`, `t=seiso-provenance`, `client=seiso`.

Schema:

- `seiso.provenance.attestation/v1` — digests without dataset merkle
- `seiso.provenance.attestation/v2` — includes `dataset_merkle_*` fields

A receipt is written back into the local sidecar:

```json
"nostr": {
  "event_id": "...",
  "pubkey": "...",
  "relays": ["wss://..."],
  "published_at": "...",
  "attestation_sha256": "..."
}
```

Later, `seiso provenance verify` recomputes local digests and fetches the event
to check the signature and digest match.

## Training-data membership (private examples)

LoRA / QLoRA / full training can commit a **Merkle root** over the **final
train-split** row fingerprints (after preprocess, `max_samples`, and
train/eval split — before tokenization).

| On relays | Kept local |
|-----------|------------|
| `dataset_merkle_root`, leaf count, alg | `dataset_merkle.json` (sorted fingerprints + tree inputs) |
| Run metadata in the attestation | Row text / labels; membership proof files |

Anyone who **holds a specific row** can prove it was in that committed corpus
without publishing the row:

```bash
seiso provenance dataset-prove path/to/seiso_manifest.json --row row.json -o proof.json
seiso provenance dataset-verify-proof proof.json --manifest path/to/seiso_manifest.json
# or: --event-id <hex> --relay wss://nos.lol
seiso provenance dataset-verify-proof proof.json --local-only
```

**This is corpus membership, not gradient provenance.** It does not prove the
sample appeared in a particular optimizer step (ZKML / step logs are out of scope).

**Privacy notes**

- Relays never receive row bodies or the fingerprint list.
- Fingerprints are SHA-256 of canonical JSON; an attacker who already has a
  candidate row from a public corpus can test membership (dictionary attack).
  For confidential corpora, treat `dataset_merkle.json` like other local secrets
  (mode `0600`) and share proofs out-of-band only with auditors.
- Disable or skip: `dataset_merkle: false` in the train YAML, or
  `SEISO_DATASET_MERKLE=0`. Large trains above
  `SEISO_DATASET_MERKLE_MAX_ROWS` (default 250000) set
  `dataset_merkle_skipped` on the manifest instead of failing the run.

## CLI (default path)

`start` enables outbound Nostr with public digests-only relays. Override or
disable as needed:

```bash
pip install 'seiso[nostr]'   # websockets + cryptography
# Defaults (also set by scripts/start.sh):
# export SEISO_ALLOW_NOSTR=1
# export SEISO_NOSTR_RELAYS=wss://nos.lol,wss://relay.damus.io

seiso provenance keygen
seiso provenance attest path/to/manifest.json
seiso provenance verify path/to/manifest.json
seiso provenance show path/to/manifest.json
seiso provenance dataset-prove path/to/seiso_manifest.json --row row.json
seiso provenance dataset-verify-proof proof.json --manifest path/to/seiso_manifest.json
```

Auto-attest after research pipelines remains opt-in (non-fatal if relays are down):

```bash
export SEISO_NOSTR_ATTEST=1
```

## Forge UI

1. First-run auth generates an npub (write it down → Continue) or imports an `nsec`.
   That key is also used for provenance signing when present.
2. Open **Integrations → Nostr provenance** to confirm relays / enable auto-attest.
3. Kill-switch: `SEISO_ALLOW_NOSTR=0` and restart Forge.

Private keys are encrypted under `$SEISO_DATA_DIR/nostr_keys/`. Auth returns a
one-time `nsec` only when Forge generated the key; settings APIs never echo `nsec`.

## Replay vs attestation

| Goal | Command / path |
|------|----------------|
| Re-check local artifact hashes | `seiso compress manifest-verify`, adaptive_quant replay CLIs |
| Check external commitment | `seiso provenance verify` |
| Prove a held row was in the train corpus | `seiso provenance dataset-prove` / `dataset-verify-proof` |

Nostr does not store checkpoints. Resume training still uses local
`resume_from` checkpoints.

## Security notes

- Default **on** (`SEISO_ALLOW_NOSTR` unset or `1`); disable with `SEISO_ALLOW_NOSTR=0`.
- Auto-attest default **off** (`SEISO_NOSTR_ATTEST` unset).
- Relays must be `wss://` (or `ws://` loopback only when explicitly allowed).
- Private / link-local / metadata addresses are blocked (SSRF policy).
- Default relays are public digests-only endpoints; override via `SEISO_NOSTR_RELAYS`.
- Event content is digests only — no dataset paths with secrets, no weights, no row text.
