---
name: seiso-orchestrate
description: >-
  Orchestrate Seiso Local AI (Forge + seiso CLI) from a generic coding agent,
  with optional Buzz channel receipts. Use when the user mentions Seiso, Forge,
  local fine-tuning, GGUF export, distill-rl, rl-quant, Nostr provenance, Buzz,
  or buzz-cli for agent-driven train/compress/export jobs.
---

# Seiso from an agent (Buzz-compatible)

Run **local** Seiso work from any agent harness (CLI loop, IDE agent, or Buzz
room). Seiso is the ML workspace
([SeisoLocalAI](https://github.com/Legendarylibr/SeisoLocalAI)).

**Buzz is optional.** When a Buzz channel is available, post receipts there for
the human+agent audit trail ([block/buzz](https://github.com/block/buzz)). The
same Seiso commands work without Buzz — only mesh multi-node requires a Buzz
agent identity.

## Training surfaces (do not mix)

| Surface | Entry | What it exposes | Mesh / multi-node |
|---------|-------|-----------------|-------------------|
| **Frontend** | Forge UI / `POST /api/training/jobs` | Full local training config (method, quant, local multi-GPU DDP `nnodes=1`, hyperparams) | **Refused** |
| **Agent** | `seiso` CLI / agent chat | Same full local config + mesh tools when opted in | Buzz-agent-only (`SEISO_ALLOW_MESH=1` + valid `BUZZ_PRIVATE_KEY` nsec). Mesh plans are NIP-01 / BIP-340 signed offline; buzz-cli does relay NIP-98. Peers also share `SEISO_MESH_TOKEN`. |

- Frontend must keep showing proper training settings — never strip the Train
  studio down to a stub when separating agent mesh.
- Mesh is **not functional yet — do not use** for real multi-node until declared
  ready. Self-hosted single-node remains free and does not need mesh flags.

Query the surface from Forge: `GET /api/training/surface`.

## Preconditions

```bash
# Seiso install (default paths)
export SEISO_INSTALL_DIR="${SEISO_INSTALL_DIR:-$HOME/Seiso}"
export SEISO_DATA_DIR="${SEISO_DATA_DIR:-$HOME/.seiso}"
export SEISO_FORGE_URL="${SEISO_FORGE_URL:-http://127.0.0.1:8765}"

# Mark this process as an agent surface (generic; not Buzz-specific)
export SEISO_AGENT=1
# or: export SEISO_TRAINING_SURFACE=agent

# Optional — Buzz room + mesh identity
# export BUZZ_PRIVATE_KEY="nsec1…"          # never paste into Seiso chat
# export BUZZ_RELAY_URL="${BUZZ_RELAY_URL:-http://localhost:3000}"

# Optional remote marketplace (pay only for someone else's GPU — never for local)
# export SEISO_PAY_URL="https://pay.example.com"
# export SEISO_PAY_TOKEN="seiso_pay_…"   # after session create; never post to Buzz

source "$SEISO_INSTALL_DIR/.venv/bin/activate"
```

Hard rules:

- Prefer `start` / `seiso` / `scripts/doctor.sh` over raw `python …`.
- Never delete `$SEISO_DATA_DIR` or its subdirs.
- Never put Seiso `nsec` / HF tokens / `SEISO_PAY_TOKEN` / `SEISO_MESH_TOKEN` in
  channel messages; post `npub` + job receipts only.
- **Self-hosted is always free** (local Forge/CLI). Do not use the sats
  marketplace against localhost.
- **Never automate Forge keygen.** Do not `POST /api/auth/register` with
  `generate`, drive the AuthPage, or scrape `nsec1…` / `ncryptsec1…`.
- Default Forge bind is localhost. Do not enable `SEISO_ALLOW_REMOTE` unless
  the operator asks.
- Smoke configs first (`configs/*_smoke.*`) before long GPU jobs.

## Room contract (when Buzz is present)

1. Join or create a channel for the run (one job family per channel when possible).
2. Set topic/purpose to the goal + config path.
3. After each milestone, `buzz messages send` a short status with receipts.
4. On failure, post the error tail + next diagnostic command; do not silently retry forever.

Without Buzz, write the same receipt JSON to the terminal / work log.

```bash
CHANNEL="<uuid>"   # from buzz channels list|create

buzz channels topic --channel "$CHANNEL" --topic "Seiso: train configs/example_lora.yaml"
buzz messages send --channel "$CHANNEL" --content "Starting doctor + Forge health check"
```

## Orchestration loop

```
- [ ] 1. Resolve env + activate Seiso venv (+ SEISO_AGENT=1)
- [ ] 2. seiso doctor (and --network if Hub needed)
- [ ] 3. Ensure Forge up (health) or start it
- [ ] 4. Confirm auth posture (npub identity; do not reset session unless asked)
- [ ] 5. Run the Seiso job (CLI smoke → real config; prefer CLI over Forge HTTP from agents)
- [ ] 6. Stream/poll until terminal status
- [ ] 7. Optional: provenance attest + verify; post event_id
- [ ] 8. Summarize artifacts under SEISO_DATA_DIR (Buzz channel or work log)
```

### Compute decision (do not charge local)

```
local Forge/CLI healthy? → self-hosted (free, 0% fee)
else SEISO_PAY_URL set? → marketplace quote → fund → job|/v1 (sats + protocol fee)
else mesh peers enough + SEISO_ALLOW_MESH + Buzz agent? → seiso mesh (experimental, no fee)
else → ask human
```

| Goal | Command / API |
|------|----------------|
| Train (LoRA/QLoRA/slime) | `seiso train --config configs/…yaml` |
| Compress | `seiso compress --config …` |
| Distill-RL | `seiso distill-rl --config …` |
| RL quant | `seiso rl-quant --config …` |
| Export / GGUF | `seiso export …` |
| Chat smoke (local free) | `seiso chat …` or `POST $SEISO_FORGE_URL/v1/chat/completions` |
| Remote marketplace | `seiso pay quote\|session\|job …` |
| Mesh (experimental, Buzz-agent-only) | `seiso mesh announce\|plan\|worker` |
| Provenance | `seiso provenance attest\|verify path/to/manifest.json` |
| Frontend surface caps | `GET $SEISO_FORGE_URL/api/training/surface` |

**Prefer CLI for agent loops** (JSON-friendly, no CSRF). Prefer Forge API /
UI when the human is watching — that path stays on the **frontend** surface
(full local training config, no mesh).

Smoke first:

```bash
seiso train --config configs/smoke_train_cpu.yaml
```

## Safety boundaries

- Seiso ≠ Buzz relay. Do not point Seiso provenance at the Buzz relay unless the
  operator explicitly allowlists that `wss://` URL.
- No DMs of secrets. No posting `nsec`, `ncryptsec`, HF tokens, cookie headers,
  or mesh tokens to channels.
- Buzz agent key (`BUZZ_PRIVATE_KEY`) and Forge instance `nsec` are different
  secrets — never reuse one as the other.
- Mesh / multi-node from the Forge UI is refused by design.

## Additional resources

- Seiso entrypoints + paths: [AGENTS.md](../../../AGENTS.md)
- CLI map: [docs/cli.md](../../../docs/cli.md)
- Forge auth + API: [docs/forge.md](../../../docs/forge.md)
- Mesh: [docs/training/mesh.md](../../../docs/training/mesh.md)
- Provenance: [docs/provenance-nostr.md](../../../docs/provenance-nostr.md)
- Buzz CLI (JSON I/O): https://github.com/block/buzz/tree/main/crates/buzz-cli
- Command cookbook: [reference.md](reference.md)

## Install into a Buzz (or other agent) checkout

```bash
# From a Seiso clone:
mkdir -p /path/to/agent-home/.agents/skills
cp -R .agents/skills/seiso-orchestrate /path/to/agent-home/.agents/skills/
# Optional Buzz mirrors:
# cp -R .agents/skills/seiso-orchestrate /path/to/buzz/.agents/skills/
```
