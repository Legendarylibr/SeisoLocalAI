# Seiso ↔ Buzz command cookbook

Companion to [SKILL.md](SKILL.md). Agent-oriented snippets only.

## Buzz channel bootstrap

```bash
export BUZZ_PRIVATE_KEY="nsec1…"
export BUZZ_RELAY_URL="${BUZZ_RELAY_URL:-http://localhost:3000}"

buzz users get
buzz channels create --name "seiso-runs" --type stream --visibility open
CHANNEL=$(buzz channels list | jq -r '.[] | select(.name=="seiso-runs") | .id' | head -1)
buzz channels join --channel "$CHANNEL"
buzz canvas set --channel "$CHANNEL" --content "$(cat <<'EOF'
# Seiso run board
- Goal:
- Config:
- Forge: http://127.0.0.1:8765
- Data: ~/.seiso
- Provenance event:
EOF
)"
```

## Seiso doctor + Forge

```bash
source "${SEISO_INSTALL_DIR:-$HOME/Seiso}/.venv/bin/activate"
seiso doctor
seiso doctor --network   # Hub reachability

curl -sS -D- -o /dev/null "$SEISO_FORGE_URL/health" | head -n1
curl -sS "$SEISO_FORGE_URL/api/auth/status" | jq .
```

Start Forge headless if down:

```bash
cd "${SEISO_INSTALL_DIR:-$HOME/Seiso}"
SEISO_NO_OPEN=1 seiso forge --no-open
```

## CLI job patterns

```bash
# Train
seiso train --config configs/smoke_train_cpu.yaml
seiso train --config configs/example_lora.yaml

# Compress / distill (research)
seiso compress --help
seiso distill-rl --help

# Export
seiso export --help

# Provenance (digests only)
seiso provenance show path/to/manifest.json
seiso provenance attest path/to/manifest.json
seiso provenance verify path/to/manifest.json
```

## Compat chat smoke (no browser)

```bash
KEY=$(cat "${SEISO_DATA_DIR:-$HOME/.seiso}/.inference_api_key" 2>/dev/null || true)
curl -sS "$SEISO_FORGE_URL/v1/chat/completions" \
  -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"user","content":"ping"}]}' | jq .
```

Never send `$KEY` to Buzz.

## Buzz status templates

**Running**

```text
## Seiso update
- job: train
- config: configs/example_lora.yaml
- status: running
- forge: http://127.0.0.1:8765
- note: <one line>
```

**Verified provenance**

```text
## Provenance receipt
- manifest: ~/.seiso/compress/.../manifest.json
- event_id: <64-hex>
- npub: npub1…
- relays: [wss://nos.lol, wss://relay.damus.io]
- verify: seiso provenance verify <manifest>
```

**Paid marketplace** (remote only — never for localhost)

```text
## Seiso marketplace run
- mode: paid
- type: finetune|slime|distill_rl|nemo_rl|inference
- status: completed|failed
- compute_sats: …
- protocol_fee_sats: …
- total_sats: …
- job_id: …
```

**Failed**

```text
## Seiso failed
- step: <name>
- exit: <code>
- hint: seiso doctor | free VRAM in Forge | check config
- log_tail:
```

```
<paste ≤40 lines>
```

## What external verification means

| Claim | Where |
|-------|--------|
| Digests committed by npub at time T | Nostr relays + `seiso provenance verify` |
| Weights / dataset integrity | Local paths / HF — not on relays |
| Buzz discussion / approvals | Buzz event log (`buzz messages *`) |
| Forge login session | Local JWT only — not federated via Buzz |

## Env cheat sheet

| Variable | Role |
|----------|------|
| `BUZZ_PRIVATE_KEY` | Agent nsec for buzz-cli (NIP-98) |
| `BUZZ_RELAY_URL` | Buzz relay HTTP(S) base |
| `SEISO_INSTALL_DIR` | Clone / install root (default `~/Seiso`) |
| `SEISO_DATA_DIR` | User data (default `~/.seiso`) |
| `SEISO_FORGE_URL` | Forge base (default `http://127.0.0.1:8765`) |
| `SEISO_ALLOW_NOSTR` | Provenance outbound gate (`0` kills) |
| `SEISO_NOSTR_RELAYS` | Comma-separated `wss://` relays |
| `SEISO_NOSTR_ATTEST` | Auto-attest after pipelines (opt-in) |
| `SEISO_INFERENCE_API_KEY` | Compat `/v1` chat key override |
| `SEISO_PAY_URL` | Remote marketplace base (never required for local) |
| `SEISO_PAY_TOKEN` | Prepaid `seiso_pay_*` session token (never post to Buzz) |
| `SEISO_ALLOW_PAY` | Operator: enable pay sidecar / CLI (opt-in) |
| `SEISO_PROTOCOL_FEE_BPS` | Protocol fee basis points (default 500 = 5%) |
| `SEISO_ALLOW_MESH` | Experimental Buzz mesh (opt-in; no protocol fee) |
| `SEISO_MESH_TOKEN` | Mesh shared secret (out-of-band; never post to Buzz) |
