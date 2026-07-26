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

The event is kind **`31250`** (parameterized replaceable) with tags
`d=<pipeline>:<run_id>`, `t=seiso-provenance`, `client=seiso`.

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
```

Auto-attest after research pipelines remains opt-in (non-fatal if relays are down):

```bash
export SEISO_NOSTR_ATTEST=1
```

## Forge UI

1. First-run onboarding generates an npub (or import an nsec).
2. Open **Integrations → Nostr provenance** to confirm relays / enable auto-attest.
3. Kill-switch: `SEISO_ALLOW_NOSTR=0` and restart Forge.

Private keys are encrypted under `$SEISO_DATA_DIR/nostr_keys/` (never returned
as `nsec` from the API).

## Replay vs attestation

| Goal | Command / path |
|------|----------------|
| Re-check local artifact hashes | `seiso compress manifest-verify`, adaptive_quant replay CLIs |
| Check external commitment | `seiso provenance verify` |

Nostr does not store checkpoints. Resume training still uses local
`resume_from` checkpoints.

## Security notes

- Default **on** (`SEISO_ALLOW_NOSTR` unset or `1`); disable with `SEISO_ALLOW_NOSTR=0`.
- Auto-attest default **off** (`SEISO_NOSTR_ATTEST` unset).
- Relays must be `wss://` (or `ws://` loopback only when explicitly allowed).
- Private / link-local / metadata addresses are blocked (SSRF policy).
- Default relays are public digests-only endpoints; override via `SEISO_NOSTR_RELAYS`.
- Event content is digests only — no dataset paths with secrets, no weights.
