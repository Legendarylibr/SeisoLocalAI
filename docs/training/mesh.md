# Experimental Buzz mesh training

Opt-in multi-node coordination via a Buzz channel. **Not** a marketplace — no protocol fee.

```bash
export SEISO_ALLOW_MESH=1
export SEISO_MESH_TOKEN="shared-out-of-band-secret"   # never post to Buzz
export BUZZ_PRIVATE_KEY=nsec1…
export BUZZ_RELAY_URL=…

seiso mesh announce --channel "$CHANNEL" --gpus 2 --capabilities finetune,slime
# → post buzz_receipt to the channel

seiso mesh plan --channel "$CHANNEL" --type finetune --nodes 2 --master-addr 10.0.0.1
# → share plan buzz_receipt; each node:

seiso mesh worker --plan "$PLAN_PATH" --rank 0
# → apply train_config_overlay / env to Accelerate multi-node launch
```

Requires reachable master (LAN/VPN/tailnet) and Seiso’s existing `distributed_*` knobs ([multi-gpu.md](multi-gpu.md)).

If the mesh lacks GPUs, fall back to a bookmarked paid marketplace URL ([pay/marketplace.md](../pay/marketplace.md)) or ask a human — do not invent cloud.
