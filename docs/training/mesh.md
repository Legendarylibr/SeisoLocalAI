# Experimental Buzz-agent mesh training

> **Secondary / opt-in path.** Local single-node Forge/CLI training remains the
> primary path. Mesh is Buzz-agent-only coordination for multi-node Accelerate
> jobs (`SEISO_ALLOW_MESH=1`). Forge UI refuses `nnodes>1`.

Opt-in **shared / multi-node** coordination via a [Buzz](https://github.com/block/buzz) **agent** identity. Peers announce capacity, agree a signed plan, import it, claim a rank, materialize a train YAML from the plan overlay, and optionally launch `seiso train`.

**Not** a marketplace — **no** protocol fee. Requires:

1. `SEISO_ALLOW_MESH=1`
2. A valid Buzz agent `BUZZ_PRIVATE_KEY` **nsec** (BIP-340 signing). `BUZZ_AUTH_TAG` alone cannot sign mesh plans in Seiso. Seiso does **not** NIP-98 to the Buzz relay — buzz-cli does that separately; Seiso signs local NIP-01 mesh events offline.
3. A shared out-of-band `SEISO_MESH_TOKEN` (≥16 chars; never post the token or plan JSON to Buzz)
4. Optional allowlist: `SEISO_MESH_TRUSTED_NPUBS` / `SEISO_MESH_TRUSTED_PUBKEYS` (comma-separated) so workers only accept plans from known agent keys
5. Optional single-host smoke: `SEISO_MESH_ALLOW_LOOPBACK=1` (permits `127.0.0.1` master — never the default)

**Forge UI / frontend training cannot start mesh.** The Train studio keeps full **local** training config (including local multi-GPU DDP with `nnodes=1`); multi-node is agent-only. See `GET /api/training/surface`.

## When to use mesh vs pay vs local

From the agent orchestration skill (local-first, Buzz-compatible):

```
local Forge/CLI healthy?     → self-hosted (free)
else mesh peers enough + Buzz agent + SEISO_ALLOW_MESH? → seiso mesh (secondary, no fee)
else SEISO_PAY_URL set?      → sats marketplace (Ark + protocol fee; not for real funds yet)
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
# → plan JSON under ~/.seiso/mesh/plans/ + signed nostr_event
#    gpus-per-node pins distributed_nproc_per_node on every worker

# Peers: import the signed event from Buzz (kind-9 content), then join
jq -c .nostr_event <plan.json | buzz messages send --channel "$CHANNEL" --content -
# on each rank machine:
seiso mesh import-plan --event plan_event.json
seiso mesh worker --plan "$JOB_ID" --rank 0 \
  --base-config configs/smoke_train_gpu.yaml --dry-run
seiso mesh worker --plan "$JOB_ID" --rank 1 \
  --base-config configs/smoke_train_gpu.yaml --launch
```

`--dry-run` materializes the worker YAML and prints the `seiso train` command without starting GPUs. `--launch` claims the rank, writes `mesh/plans/<job_id>/rank-<n>-train.yaml`, and runs train.

`protocol_fee_sats` on plans is always `0`; `market` is `false`.

Plans are sandboxed under `mesh/plans/<job_id>.json` — absolute foreign paths are refused. Each plan is a **NIP-01** event (kind `31251`) signed with the agent nsec (**BIP-340 Schnorr**). Workers verify the signature + body match, then check the shared `SEISO_MESH_TOKEN` HMAC (bound to job id + planner pubkey).

## Frontend vs agent

| | Frontend (Forge UI) | Agent (CLI / Buzz chat) |
|--|---------------------|-------------------------|
| Local training config | Full (method, quant, DDP `nnodes=1`, hyperparams) | Full |
| Multi-node / mesh | Refused | Opt-in secondary (Nostr-signed) |

## Requirements

- Reachable master (`distributed_master_addr`) on LAN / VPN / tailnet — not `127.0.0.1` when `nodes>=2` unless `SEISO_MESH_ALLOW_LOOPBACK=1`
- Seiso multi-node knobs: [multi-gpu.md](multi-gpu.md)
- Trusted collaborators only — peers bind via (1) Nostr signature from the planner nsec and (2) shared out-of-band `SEISO_MESH_TOKEN`. Optional pubkey allowlist via `SEISO_MESH_TRUSTED_NPUBS`.
- Real multi-host jobs need GPUs on each rank and a reachable master; CPU dry-run covers materialize/command wiring only.

## Buzz / agent receipts (safe to post)

**Relay only with signing.** Channel authority is the signed NIP-01 `nostr_event`
(BIP-340). Unsigned `buzz_receipt` / `agent_receipt` JSON is a local pointer
(`npub` + `event_id`) — do not treat it as proof by itself.

```json
{"id":"…","pubkey":"…","created_at":0,"kind":31251,"tags":[…],"content":"{…}","sig":"…"}
```

**Buzz transport:** Buzz channel messages only accept kinds `9` / `45001` / `45003`
(`buzz messages send --kind`); `buzz social publish` is kind `1` only. Do **not**
pass `--kind 31251` (rejected). Embed the signed event JSON as message *content*
so peers can extract and `verify_event` offline:

```bash
# after seiso mesh plan / announce / worker / seiso agent status
jq -c .nostr_event <out.json | buzz messages send --channel "$CHANNEL" --content -
```

On a general Nostr relay you may publish the same event as a native addressable
EVENT. Never post `SEISO_MESH_TOKEN`, local `token_fingerprint` (HMAC stays on
disk only — not in signed event content), nsecs, or unsigned plan JSON as authority.

**Privacy:** announce defaults to an opaque `peer-<fingerprint>` alias — OS
hostname stays on the local disk record and is never signed. Do not pass
`--alias` equal to your machine hostname (signing refuses). Plan
`distributed_master_addr` is intentionally in the signed event (peers need it);
treat channel posts of plans as LAN-topology-sensitive and prefer private
channels / trusted peers.

## Fallback

If the mesh lacks GPUs, fall back to a bookmarked paid marketplace URL ([pay/marketplace.md](../pay/marketplace.md)) or ask a human — do not invent cloud.
