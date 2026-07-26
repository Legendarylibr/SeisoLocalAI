# Nostr provenance attestation

Seiso can **optionally** publish a small signed commitment of a run’s local
manifest digests to Nostr relays. Weights and full manifests stay on disk (or
Hugging Face). This does **not** resume training and is **not** a social client.

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

## Enable (CLI)

```bash
pip install 'seiso[nostr]'   # websockets + cryptography
export SEISO_ALLOW_NOSTR=1
export SEISO_NOSTR_RELAYS=wss://relay.example.com

seiso provenance keygen
seiso provenance attest path/to/manifest.json
seiso provenance verify path/to/manifest.json
seiso provenance show path/to/manifest.json
```

Auto-attest after research pipelines (non-fatal if relays are down):

```bash
export SEISO_NOSTR_ATTEST=1
```

## Enable (Forge UI)

1. Set `SEISO_ALLOW_NOSTR=1` and restart Forge.
2. Open **Integrations → Nostr provenance**.
3. Generate or import a key, set allowlisted `wss://` relays, optionally enable
   auto-attest for completed pipeline / export / RL-quant jobs.

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

- Default **off** (`SEISO_ALLOW_NOSTR` unset).
- Relays must be `wss://` (or `ws://` loopback only when explicitly allowed).
- Private / link-local / metadata addresses are blocked (SSRF policy).
- Empty relay allowlist means no outbound Nostr.
- Event content is digests only — no dataset paths with secrets, no weights.
