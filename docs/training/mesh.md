# Experimental Buzz mesh training

> **Not functional yet — do not use.** Mesh coordination is experimental scaffolding. Do not rely on it for real multi-node jobs until it is declared ready.

Opt-in **shared / multi-node** coordination via a [Buzz](https://github.com/block/buzz) channel. Peers announce capacity, agree a plan, and apply Seiso’s existing Accelerate `distributed_*` knobs.

**Not** a marketplace — **no** protocol fee. Requires `SEISO_ALLOW_MESH=1` and a shared out-of-band `SEISO_MESH_TOKEN` (never post the token to Buzz).

## When to use mesh vs pay vs local

From the Buzz orchestration skill (local-first):

```
local Forge/CLI healthy?     → self-hosted (free)
else mesh peers enough?      → seiso mesh (experimental, no fee)
else SEISO_PAY_URL set?      → sats marketplace (Ark + protocol fee)
else                         → ask a human
```

See [pay/marketplace.md](../pay/marketplace.md) and [`.agents/skills/seiso-orchestrate/SKILL.md`](../../.agents/skills/seiso-orchestrate/SKILL.md).

## Flow

```bash
export SEISO_ALLOW_MESH=1
export SEISO_MESH_TOKEN="shared-out-of-band-secret"   # never post to Buzz
export BUZZ_PRIVATE_KEY=nsec1…                        # Buzz agent / CLI identity
export BUZZ_RELAY_URL=…

# Each machine with spare GPUs
seiso mesh announce --channel "$CHANNEL" --gpus 2 --capabilities finetune,slime
# → buzz_receipt { role: announce, alias, gpus, mesh_endpoint_fingerprint }
#    post that JSON to the Buzz channel (no secrets)

# Planner (any trusted peer)
seiso mesh plan --channel "$CHANNEL" --type finetune --nodes 2 --master-addr 10.0.0.1
# → plan JSON on disk + buzz_receipt { role: plan, job_id, world_size, … }
#    share the plan receipt; distribute plan_path / job_id out-of-band if needed

# Each rank
seiso mesh worker --plan "$PLAN_PATH" --rank 0
# → env + train_config_overlay for Accelerate; buzz_receipt { role: heartbeat, status: joining }
#    apply overlay to train YAML / accelerate launch; post heartbeat receipt
```

`protocol_fee_sats` on plans is always `0`; `market` is `false`.

## Requirements

- Reachable master (`distributed_master_addr`) on LAN / VPN / tailnet
- Seiso multi-node knobs: [multi-gpu.md](multi-gpu.md)
- Trusted collaborators only — mesh does **not** authenticate arbitrary Buzz posters cryptographically against Seiso; trust is “have the OOB token + follow receipts”

## Buzz receipts (safe to post)

```json
{"role":"announce","channel":"…","gpus":2,"capabilities":["finetune","slime"],"alias":"node-a","mesh_endpoint_fingerprint":"…"}
```

```json
{"role":"plan","job_id":"…","type":"finetune","world_size":2,"status":"planned","master_hint":"10.0.0.1"}
```

```json
{"role":"heartbeat","job_id":"…","type":"finetune","rank":0,"world_size":2,"status":"joining"}
```

Never post `SEISO_MESH_TOKEN`, private IPs you consider sensitive beyond the agreed master hint, dataset paths, or nsecs.

## Fallback

If the mesh lacks GPUs, fall back to a bookmarked paid marketplace URL ([pay/marketplace.md](../pay/marketplace.md)) or ask a human — do not invent cloud.
