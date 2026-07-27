---
name: seiso-orchestrate
description: >-
  Orchestrate Seiso Local AI (Forge + seiso CLI) from a Buzz agent room using
  buzz-cli for channel updates and Seiso for local train/compress/export/
  provenance jobs. Use when the user mentions Buzz, buzz-cli, Seiso, Forge,
  local fine-tuning, GGUF export, distill-rl, rl-quant, or Nostr provenance
  attest/verify from an agent channel.
---

# Seiso from Buzz

Run **local** Seiso work from a Buzz channel. Buzz is the room + audit trail
([block/buzz](https://github.com/block/buzz)); Seiso is the ML workspace
([SeisoLocalAI](https://github.com/Legendarylibr/SeisoLocalAI)).

Humans and agents share the Buzz room. Seiso stays on the machine (localhost).
Post **receipts** (commands, job ids, manifest paths, Nostr `event_id`) back to
the channel — not vibes.

## Preconditions

```bash
# Buzz agent identity (required for buzz-cli)
export BUZZ_PRIVATE_KEY="nsec1…"          # agent key — never paste into Seiso chat
export BUZZ_RELAY_URL="${BUZZ_RELAY_URL:-http://localhost:3000}"

# Seiso install (default paths)
export SEISO_INSTALL_DIR="${SEISO_INSTALL_DIR:-$HOME/Seiso}"
export SEISO_DATA_DIR="${SEISO_DATA_DIR:-$HOME/.seiso}"
export SEISO_FORGE_URL="${SEISO_FORGE_URL:-http://127.0.0.1:8765}"

# Activate Seiso venv before every seiso command
source "$SEISO_INSTALL_DIR/.venv/bin/activate"
```

Hard rules:

- Prefer `start` / `seiso` / `scripts/doctor.sh` over raw `python …`.
- Never delete `$SEISO_DATA_DIR` or its subdirs.
- Never put Seiso `nsec` / HF tokens in Buzz messages; post `npub` + job receipts only.
- **Never automate Forge keygen.** Do not `POST /api/auth/register` with `generate`, drive the AuthPage, click **Download encrypted .txt**, read `seiso-ncryptsec-backup.txt`, collect backup passphrases, or scrape `sessionStorage` / DOM for `nsec1…` / `ncryptsec1…`. Onboarding backup is human-only.
- Default Forge bind is localhost. Do not enable `SEISO_ALLOW_REMOTE` unless the operator asks.
- Smoke configs first (`configs/*_smoke.*`) before long GPU jobs.

## Room contract

1. Join or create a channel for the run (one job family per channel when possible).
2. Set topic/purpose to the goal + config path.
3. After each milestone, `buzz messages send` a short status with receipts.
4. On failure, post the error tail + next diagnostic command; do not silently retry forever.

```bash
CHANNEL="<uuid>"   # from buzz channels list|create

buzz channels topic --channel "$CHANNEL" --topic "Seiso: train configs/example_lora.yaml"
buzz messages send --channel "$CHANNEL" --content "Starting doctor + Forge health check"
```

## Orchestration loop

Copy and tick:

```
- [ ] 1. Resolve env + activate Seiso venv
- [ ] 2. seiso doctor (and --network if Hub needed)
- [ ] 3. Ensure Forge up (health) or start it
- [ ] 4. Confirm auth posture (npub identity; do not reset session unless asked)
- [ ] 5. Run the Seiso job (CLI smoke → real config, or Forge API job)
- [ ] 6. Stream/poll until terminal status
- [ ] 7. Optional: provenance attest + verify; post event_id
- [ ] 8. Summarize artifacts under SEISO_DATA_DIR into the Buzz channel
```

### 1–3 · Health

```bash
seiso doctor
curl -sf "$SEISO_FORGE_URL/health" || (cd "$SEISO_INSTALL_DIR" && seiso forge --no-open &)
# wait until /health is 200
```

Post to Buzz: doctor summary (ok/warn) + Forge URL.

### 4 · Auth (local only)

Forge auth is Nostr npub/nsec (no passwords). Agents use an already-onboarded
instance or the inference API key — **never** interactive browser keygen or the
onboarding **Download encrypted .txt** backup (NIP-49 `ncryptsec`; still human-only,
and never post the passphrase or file contents to Buzz).

```bash
# Session status (no secrets) — safe
curl -sf "$SEISO_FORGE_URL/api/auth/status"

# Forbidden for agents (returns or creates nsec):
# curl -X POST "$SEISO_FORGE_URL/api/auth/register" -d '{"generate":true}'
# curl -X POST "$SEISO_FORGE_URL/api/auth/login" -d '{"nsec":"..."}'

# Compat chat-only key (if configured) — never post the key value to Buzz
# ~/.seiso/.inference_api_key or SEISO_INFERENCE_API_KEY
```

If onboarding is required, stop and ask the human to open `$SEISO_FORGE_URL`,
generate a key, back up the **nsec** offline (handwrite and/or NIP-49 encrypted
download — keep passphrase + file out of Buzz), Continue — then resume.
Post only the **npub** (public identity) to the channel if needed.

### 5 · Run jobs

**Prefer CLI for agent loops** (JSON-friendly, no CSRF). Prefer Forge API when the
human is watching the UI / SSE.

| Goal | Command / API |
|------|----------------|
| Train (LoRA/QLoRA/slime) | `seiso train --config configs/…yaml` |
| Compress | `seiso compress --config …` |
| Distill-RL | `seiso distill-rl --config …` |
| RL quant | `seiso rl-quant --config …` |
| Export / GGUF | `seiso export …` |
| Chat smoke | `seiso chat …` or `POST $SEISO_FORGE_URL/v1/chat/completions` |
| Provenance | `seiso provenance attest\|verify path/to/manifest.json` |

Smoke first:

```bash
seiso train --config configs/smoke_train_cpu.yaml
```

Forge training job (needs session cookie/CSRF or use CLI instead):

```bash
# Prefer CLI from agents. If using HTTP, follow Forge CSRF + cookie rules in docs/forge.md.
curl -sf "$SEISO_FORGE_URL/api/training/jobs"
```

### 6 · Progress in the room

```bash
buzz messages send --channel "$CHANNEL" --content "$(cat <<'EOF'
## Seiso update
- step: train
- config: configs/example_lora.yaml
- status: running
- log_tail: <last useful lines>
EOF
)"
```

On completion, include artifact paths relative to `$SEISO_DATA_DIR` (checkpoints,
exports, manifests). Do not attach weights into Buzz.

### 7 · External verification (digests only)

Seiso can seal **manifest digests** to Nostr relays (not weights). Anyone with
`event_id` + relays can verify the signature and digest commitment.

```bash
# Outbound default-on; kill with SEISO_ALLOW_NOSTR=0
seiso provenance attest path/to/manifest.json
seiso provenance verify path/to/manifest.json
seiso provenance show path/to/manifest.json
```

Post to Buzz: `event_id`, `npub`, relay list, `attestation_sha256`, manifest path.
That is the external receipt. Local hash-verify / re-run stays on the machine.

### 8 · Close the loop

```bash
buzz messages send --channel "$CHANNEL" --content "$(cat <<'EOF'
## Seiso done
- result: success|failed
- artifacts: ~/.seiso/...
- provenance_event_id: <hex or none>
- next: <human decision>
EOF
)"
buzz reactions add --event "<status-event-id>" --emoji "✅"   # or "❌"
```

## Decision guide

| Ask | Do |
|-----|----|
| Quick iteration / CI agent | `configs/*_smoke.*` via CLI |
| Human watching UI | `seiso forge` + Forge pages; still post Buzz receipts |
| Need external digest commitment | `seiso provenance attest` then post `event_id` |
| Hub / gated models | Human adds HF token in Settings; agent runs `seiso doctor --network` |
| Lost Forge nsec | Human must "start a new session"; agent must not invent credentials |

## Safety boundaries

- Seiso ≠ Buzz relay. Do not point Seiso provenance at the Buzz relay unless the
  operator explicitly allowlists that `wss://` URL and understands digests-only.
- No DMs of secrets. No posting `nsec`, `ncryptsec`, backup passphrases, HF tokens,
  cookie headers, AuthPage HTML, `seiso-ncryptsec-backup.txt` contents, or
  register/login response bodies to channels.
- Buzz agent key (`BUZZ_PRIVATE_KEY`) and Forge instance `nsec` are different
  secrets — never reuse one as the other; never paste either into Seiso chat.
- GPU jobs: call memory-guard / unload patterns when docs say so; don't stack
  heavy jobs without checking VRAM (`seiso doctor`, Forge metrics).
- Remote tools / code-exec stay opt-in per Seiso security docs — do not enable
  from Buzz orchestration unless requested.

## Additional resources

- Seiso entrypoints + paths: [AGENTS.md](../../../AGENTS.md)
- CLI map: [docs/cli.md](../../../docs/cli.md)
- Forge auth + API: [docs/forge.md](../../../docs/forge.md)
- Provenance: [docs/provenance-nostr.md](../../../docs/provenance-nostr.md)
- Buzz CLI (JSON I/O): https://github.com/block/buzz/tree/main/crates/buzz-cli
- Command cookbook: [reference.md](reference.md)

## Install into a Buzz checkout

```bash
# From a Seiso clone:
mkdir -p /path/to/buzz/.agents/skills
cp -R .agents/skills/seiso-orchestrate /path/to/buzz/.agents/skills/
# Optional mirrors Buzz uses for other harnesses:
# cp -R .agents/skills/seiso-orchestrate /path/to/buzz/.goose/skills/
```
