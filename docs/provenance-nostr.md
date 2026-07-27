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

### Why this exists

A signed Nostr attestation proves “this run sealed these digests.” It does
**not** by itself tell an auditor that *their* held example was in the training
set. Publishing every row would leak private data.

So training also commits a **Merkle root** over fingerprints of the **final
train-split** rows. Relays see only the root. Anyone who already holds a
specific row can build a short path proof and show it was in that committed
corpus — without putting the row text on Nostr.

### Plain picture

1. Train finishes preprocess / `max_samples` / train–eval split (before tokenize).
2. Seiso fingerprints each train row (canonical JSON → SHA-256), sorts unique
   fingerprints, and builds a Merkle tree (domain-separated leaves/nodes).
3. Local `dataset_merkle.json` keeps the fingerprint list (mode `0600`).
4. `seiso_manifest.json` records `dataset_merkle_root`, leaf count, and alg.
5. Nostr attestation **v2** seals those merkle fields with the other digests.
6. Later: hold a row → `dataset-prove` → share `proof.json` →
   `dataset-verify-proof` checks the path and (optionally) the sealed root.

| On relays | Kept local |
|-----------|------------|
| `dataset_merkle_root`, leaf count, alg | `dataset_merkle.json` (sorted fingerprints) |
| Run metadata in the attestation | Row text / labels; membership proof files |

### What it is / is not

| Proves | Does **not** prove |
|--------|--------------------|
| A held row’s fingerprint was a leaf under the sealed root for that `run_id` | The sample appeared in a particular optimizer step (no ZKML / step logs) |
| Attestation signature + digest match (when verifying via Nostr) | Model weights, gradients, or “fair” training |

Corpus membership ≠ gradient provenance.

### End-to-end CLI

```bash
# After a train that wrote seiso_manifest.json + dataset_merkle.json:
seiso provenance attest path/to/seiso_manifest.json

# Auditor (or you) holds a candidate row as JSON object matching train columns:
seiso provenance dataset-prove path/to/seiso_manifest.json \
  --row row.json -o proof.json

# Check merkle path + that root matches the Nostr attestation on the manifest:
seiso provenance dataset-verify-proof proof.json \
  --manifest path/to/seiso_manifest.json

# Or fetch a known event id from relays:
seiso provenance dataset-verify-proof proof.json \
  --event-id <hex> --relay wss://nos.lol

# Offline: only recompute path vs root embedded in the proof
seiso provenance dataset-verify-proof proof.json --local-only
```

`dataset-prove` needs the local sidecar next to the manifest
(`dataset_merkle.json`). Share proofs out-of-band; do not publish the sidecar.

### Config and skip rules

| Control | Effect |
|---------|--------|
| `dataset_merkle: true` (default) in train YAML | Commit merkle for the run |
| `dataset_merkle: false` | Skip commit |
| `SEISO_DATASET_MERKLE=0` | Env override off (also `1`/`true` to force on) |
| `SEISO_DATASET_MERKLE_MAX_ROWS` (default `250000`) | Above this, set `dataset_merkle_skipped` on the manifest; train still succeeds |

Fingerprint timing: **after** preprocess, dedupe, `max_samples`, and the
train/eval split; **before** tokenization. Eval-only rows are not in the tree.

### Privacy notes

- Relays never receive row bodies or the fingerprint list.
- Fingerprints are SHA-256 of canonical JSON; an attacker who already has a
  candidate row from a public corpus can test membership (dictionary attack).
  For confidential corpora, treat `dataset_merkle.json` like other local secrets
  and share proofs only with intended auditors.
- Event content remains digests only — no dataset paths with secrets, no weights.

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

1. First-run auth generates a key (write down the one-time `nsec` → Continue; `npub`
   is shown as the public identity) or imports an `nsec`.
   That key is also used for provenance signing when present.
2. Open **Integrations → Nostr provenance** to confirm relays / enable auto-attest.
   Relays are a per-user allowlist for the account’s **npub** (digests-only publish),
   not stored on or derived from the `nsec`.
3. Kill-switch: `SEISO_ALLOW_NOSTR=0` and restart Forge.

Private keys are encrypted under `$SEISO_DATA_DIR/nostr_keys/`. Auth returns a
one-time `nsec` only when Forge generated the key (for the write-down screen);
login and settings APIs never echo `nsec`.

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
