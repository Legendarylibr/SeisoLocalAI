# Seiso sats marketplace (opt-in)

Remote **finetune / RL / inference** priced in sats. Self-hosted Seiso stays **free** and never pays a protocol fee.

## Modes

| Mode | Compute | Sats | Protocol fee |
|------|---------|------|--------------|
| Self-hosted | Your Forge / CLI | None | None |
| Marketplace | Operator `SEISO_PAY_URL` | Buyer pays quote | Default **5%** on top of compute |
| Mesh (experimental) | Buzz peers | Usually none | None |

## Operator (sell capacity)

```bash
export SEISO_ALLOW_PAY=1
export SEISO_PAY_FAUCET=1                          # dev only
export SEISO_PROTOCOL_TREASURY_ARK=ark1…           # required for non-faucet settles
export SEISO_OPERATOR_ARK=ark1…
export SEISO_PROTOCOL_FEE_BPS=500                  # 5%
source .venv/bin/activate
seiso forge --no-open &                            # localhost :8765
seiso pay serve --host 127.0.0.1 --port 8787       # sidecar; put TLS in front for public
```

Discovery: `GET /.well-known/seiso-pay.json`

Forge stays on `127.0.0.1`. Do not enable `SEISO_ALLOW_REMOTE` just for the market — expose the **pay sidecar** only.

## Buyer (Buzz agent / CLI)

```bash
export SEISO_ALLOW_PAY=1
export SEISO_PAY_FAUCET=1   # or fund via Ark address from session create
seiso pay quote --type finetune --preset smoke
# → compute_sats, protocol_fee_sats, total_sats, …

seiso pay session create --sats 20000 --scopes inference,finetune,rl
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

## Protocol fee

- Default `SEISO_PROTOCOL_FEE_BPS=500` (5%), added **on top** of operator list price.
- Quotes always show the split (`payee_operator_sats`, `payee_protocol_sats`).
- Clamped to 0–10% unless `SEISO_PROTOCOL_FEE_OVERRIDE=1`.
- Without `SEISO_PROTOCOL_TREASURY_ARK`, real Ark settles refuse (fail closed). Faucet may simulate splits for tests.

## Buzz receipts

```text
## Seiso run
- mode: paid
- type: finetune|slime|…
- compute_sats: …
- protocol_fee_sats: …
- total_sats: …
- job_id: …
```

Never post `SEISO_PAY_TOKEN`, nsecs, or HF tokens to the channel.

See also: [training/mesh.md](../training/mesh.md) (experimental, no fee).
