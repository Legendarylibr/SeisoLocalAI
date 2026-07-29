# Seiso sats marketplace (opt-in)

> **Not functional yet — do not use.** This surface is scaffolding / docs only. Do not run it for production work or real funds. Live Ark settlement is not wired; faucet/sim only for local smoke tests.

Remote **finetune / RL / inference** priced in sats. Self-hosted Seiso stays **free** and never pays a protocol fee.

Requires explicit `SEISO_ALLOW_PAY=1`. Leave unset for normal local Forge/CLI use.

## Modes

| Mode | Compute | Sats | Protocol fee |
|------|---------|------|--------------|
| Self-hosted | Your Forge / CLI | None | None |
| Marketplace | Operator `SEISO_PAY_URL` | Buyer pays quote | Default **5%** on top of compute |
| Mesh (experimental) | Buzz peers | Usually none | None |

## Operator (sell capacity)

```bash
export SEISO_ALLOW_PAY=1
export SEISO_PROTOCOL_TREASURY_ARK=ark1…           # required for non-faucet settles
export SEISO_OPERATOR_ARK=ark1…
export SEISO_PROTOCOL_FEE_BPS=500                  # 5%
# export SEISO_PAY_FAUCET=1                        # DEV ONLY — never on a public market
source .venv/bin/activate
seiso forge --no-open &                            # localhost :8765
seiso pay serve --host 127.0.0.1 --port 8787       # sidecar; put TLS in front for public
```

Discovery: `GET /.well-known/seiso-pay.json`

Forge stays on `127.0.0.1`. Do not enable `SEISO_ALLOW_REMOTE` just for the market — expose the **pay sidecar** only.

## Buyer (Buzz agent / CLI)

```bash
export SEISO_ALLOW_PAY=1
export SEISO_PAY_URL=https://pay.example.com       # operator sidecar
# Dev against a local faucet operator:
# export SEISO_PAY_FAUCET=1
seiso pay quote --type finetune --preset smoke
# → compute_sats, protocol_fee_sats, total_sats, …

seiso pay session create --sats 20000 --scopes inference,finetune,rl
# → token once; funding.ark_address for Ark pay-in (or faucet when enabled)
export SEISO_PAY_TOKEN=seiso_pay_…

# Inference (remote)
curl "$SEISO_PAY_URL/v1/chat/completions" \
  -H "Authorization: Bearer $SEISO_PAY_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"user","content":"ping"}]}'

# Jobs
seiso pay job start --type finetune --preset smoke --dry-run
seiso pay job start --type slime --preset smoke --dry-run
```

Local free path (no pay):

```bash
curl "$SEISO_FORGE_URL/v1/chat/completions" -H "Authorization: Bearer $LOCAL_KEY" …
seiso train -c configs/smoke_train_cpu.yaml
```

## Opt-in Ark settlement

Marketplace funding and fee splits are designed around **Ark** addresses (operator + protocol treasury). This is **opt-in** with the pay flag — local Seiso never needs Ark.

> **Status: not functional currently — do not use.** Live Ark pay-in / Bark–Second backend settlement is **not implemented**. Selecting `SEISO_ARK_BACKEND=bark|second|ark` errors until a client is bundled. Dev/smoke only: faucet (`SEISO_PAY_FAUCET=1`) or simulated ledger receipts. Do not run this marketplace for production or real funds.

| Variable | Role |
|----------|------|
| `SEISO_OPERATOR_ARK` | Operator receive address (buyer funding destination) |
| `SEISO_PROTOCOL_TREASURY_ARK` | Protocol fee destination; **required** for real settles (fail-closed) |
| `SEISO_ARK_NETWORK` | Default `signet` (or mainnet when you mean it) |
| `SEISO_ARK_BACKEND` | Empty / unset = simulated or faucet; `bark` / `second` **not functional yet** |
| `SEISO_PAY_FAUCET` | Dev faucet — credits sessions without chain IO; **never** enable publicly |

Behavior today (`seiso/pay/ark.py`):

1. **Faucet / simulated** — ledger-shaped receipts with operator + protocol split when faucet is on or treasury is set for simulation.
2. **Fail closed** — without `SEISO_PROTOCOL_TREASURY_ARK` and without faucet, paid settles refuse.
3. **`SEISO_ARK_BACKEND=bark|second|ark`** — **not functional currently**; raises a clear error until the Bark/Second client is installed. Prefer faucet for tests.

Quotes always show the fee split (`payee_operator_sats`, `payee_protocol_sats`).

## Protocol fee

- Default `SEISO_PROTOCOL_FEE_BPS=500` (5%), added **on top** of operator list price.
- Clamped to 0–10% unless `SEISO_PROTOCOL_FEE_OVERRIDE=1`.

## Buzz receipts

Agents orchestrating paid runs should post a short receipt to the Buzz channel (see [`.agents/skills/seiso-orchestrate/`](../../.agents/skills/seiso-orchestrate/SKILL.md)):

```text
## Seiso run
- mode: paid
- type: finetune|slime|…
- compute_sats: …
- protocol_fee_sats: …
- total_sats: …
- job_id: …
```

Never post `SEISO_PAY_TOKEN`, nsecs, HF tokens, or Ark private material to the channel.

## Security notes (operators)

- Forge is **single-tenant**; do not share one Forge UI as multiuser SaaS.
- Session create on the sidecar is open to anyone who can reach it — put TLS + rate limits in front; keep faucet off in public.
- Prefer fixed presets over buyer-supplied config paths until allowlisted.
- Details: [docs/README.md](../README.md) · multiuser posture discussed in nest research notes.

See also: [training/mesh.md](../training/mesh.md) (experimental Buzz shared training, no fee).
