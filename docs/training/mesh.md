# Experimental Buzz-agent mesh training

> **Not functional yet — do not use.** Mesh coordination is experimental scaffolding. Do not rely on it for real multi-node jobs until it is declared ready.

Opt-in **shared / multi-node** coordination via a [Buzz](https://github.com/block/buzz) **agent** identity. Peers announce capacity, agree a plan, and apply Seiso’s existing Accelerate `distributed_*` knobs.

**Not** a marketplace — **no** protocol fee. Requires:

1. `SEISO_ALLOW_MESH=1`
2. A valid Buzz agent `BUZZ_PRIVATE_KEY` **nsec** (BIP-340 signing). `BUZZ_AUTH_TAG` alone cannot sign mesh plans in Seiso. Seiso does **not** NIP-98 to the Buzz relay — buzz-cli does that separately; Seiso signs local NIP-01 mesh events offline.
3. A shared out-of-band `SEISO_MESH_TOKEN` (≥16 chars; never post the token or plan JSON to Buzz)
4. Optional allowlist: `SEISO_MESH_TRUSTED_NPUBS` / `SEISO_MESH_TRUSTED_PUBKEYS` (comma-separated) so workers only accept plans from known agent keys

**Forge UI / frontend training cannot start mesh.** The Train studio keeps full **local** training config (including local multi-GPU DDP with `nnodes=1`); multi-node is agent-only. See `GET /api/training/surface`.

## When to use mesh vs pay vs local

From the agent orchestration skill (local-first, Buzz-compatible):

```
local Forge/CLI healthy?     → self-hosted (free)
else mesh peers enough + Buzz agent + SEISO_ALLOW_MESH? → seiso mesh (experimental, no fee)
else SEISO_PAY_URL set?      → sats marketplace (Ark + protocol fee)
else                         → ask a human
```

See [pay/marketplace.md](../pay/marketplace.md) and [`.agents/skills/seiso-orchestrate/SKILL.md`](../../.agents/skills/seiso-orchestrate/SKILL.md).

## Flow

```bash
export SEISO_ALLOW_MESH=1
export SEISO_MESH_TOKEN="shared-out-of-band-secret"   # never post to Buzz
export BUZZ_PRIVATE_KEY=nsec1…                        # Buzz agent identity (required)
export BUZZ_RELAY_URL=…
export SEISO_AGENT=1                                  # generic agent surface marker

# Each machine with spare GPUs
seiso mesh announce --channel "$CHANNEL" --gpus 2 --capabilities finetune,slime
# → buzz_receipt / agent_receipt { role: announce, … }

# Planner (any trusted Buzz agent peer)
seiso mesh plan --channel "$CHANNEL" --type finetune --nodes 2 \
  --master-addr 10.0.0.1 --gpus-per-node 2
# → plan JSON under ~/.seiso/mesh/plans/ + receipts
#    gpus-per-node pins distributed_nproc_per_node on every worker

# Each rank ( --rank is required; do not omit )
seiso mesh worker --plan "$JOB_ID" --rank 0
seiso mesh worker --plan "$JOB_ID" --rank 1
# → apply train_config_overlay (not env-only NNODES) to seiso train / Accelerate
```

`protocol_fee_sats` on plans is always `0`; `market` is `false`.

Plans are sandboxed under `mesh/plans/<job_id>.json` — absolute foreign paths are refused. Each plan is a **NIP-01** event (kind `31251`) signed with the agent nsec (**BIP-340 Schnorr**). Workers verify the signature + body match, then check the shared `SEISO_MESH_TOKEN` HMAC (bound to job id + planner pubkey).

## Frontend vs agent

| | Frontend (Forge UI) | Agent (CLI / Buzz chat) |
|--|---------------------|-------------------------|
| Local training config | Full (method, quant, DDP `nnodes=1`, hyperparams) | Full |
| Multi-node / mesh | Refused | Opt-in experimental (Nostr-signed) |

## Requirements

- Reachable master (`distributed_master_addr`) on LAN / VPN / tailnet — not `127.0.0.1` when `nodes>=2`
- Seiso multi-node knobs: [multi-gpu.md](multi-gpu.md)
- Trusted collaborators only — peers bind via (1) Nostr signature from the planner nsec and (2) shared out-of-band `SEISO_MESH_TOKEN`. Optional pubkey allowlist via `SEISO_MESH_TRUSTED_NPUBS`.

## Buzz / agent receipts (safe to post)

**Relay only with signing.** Channel authority is the signed NIP-01 `nostr_event`
(BIP-340). Unsigned `buzz_receipt` / `agent_receipt` JSON is a local pointer
(`npub` + `event_id`) — do not treat it as proof by itself.

```json
{"id":"…","pubkey":"…","created_at":0,"kind":31251,"tags":[…],"content":"{…}","sig":"…"}
```

Publish that signed event via buzz-cli (or an allowlisted Nostr relay). Never post `SEISO_MESH_TOKEN`, `token_fingerprint`, nsecs, or unsigned plan JSON as authority.

## Fallback

If the mesh lacks GPUs, fall back to a bookmarked paid marketplace URL ([pay/marketplace.md](../pay/marketplace.md)) or ask a human — do not invent cloud.
