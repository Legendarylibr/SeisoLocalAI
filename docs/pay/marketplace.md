# Seiso sats marketplace (opt-in)

> **Not functional yet — do not use.** This surface is scaffolding / docs only. Do not run it for production work or real funds. Live **Ark** and **L402** settlement are not wired; faucet/sim only for local smoke tests.

Remote **finetune / RL / inference** priced in sats. Self-hosted Seiso stays **free** and never pays a protocol fee.

Requires explicit `SEISO_ALLOW_PAY=1`. Leave unset for normal local Forge/CLI use.

## Modes

| Mode | Compute | Sats | Protocol fee |
|------|---------|------|--------------|
| Self-hosted | Your Forge / CLI | None | None |
| Marketplace | Operator `SEISO_PAY_URL` | Buyer pays quote | Default **5%** on top of compute |
| Mesh (experimental) | Buzz peers | Usually none | None |

## Payment methods

Marketplace funding is designed to support multiple sats rails. **None of the live rails are functional yet — do not use for real funds.**

| Method | How it works (when wired) | Status today |
|--------|---------------------------|--------------|
| **Ark** | Buyer pays into operator Ark address; fee split to operator + protocol treasury | **Not functional** — Bark/Second client not bundled; faucet/sim only |
| **L402** ([Lightning HTTP 402](https://lightningfaucet.com/learn/l402-payments-explained/)) | Server returns `HTTP 402` + `WWW-Authenticate: L402` with a BOLT-11 invoice and macaroon; client pays Lightning, retries with `Authorization: L402 <macaroon>:<preimage>` | **Not functional** — challenge minting / invoice / preimage verify not bundled |
| **Dev faucet** | `SEISO_PAY_FAUCET=1` credits a session without chain IO | Smoke tests **only** — never on a public market |

Discovery advertises these under `payment_methods` in `GET /.well-known/seiso-pay.json` and on session `funding` payloads. Hide L402 from discovery with `SEISO_PAY_L402=0` (Ark still listed).

L402 fits agent/API buyers especially well: no accounts, machine-readable 402 challenge, sat-denominated per session or (later) per request. See the [L402 payments explained](https://lightningfaucet.com/learn/l402-payments-explained/) reference for the wire format.

## Operator (sell capacity)

```bash
export SEISO_ALLOW_PAY=1
export SEISO_PROTOCOL_TREASURY_ARK=ark1…           # required for non-faucet settles
export SEISO_OPERATOR_ARK=ark1…
export SEISO_PROTOCOL_FEE_BPS=500                  # 5%
# export SEISO_PAY_FAUCET=1                        # DEV ONLY — never on a public market
# export SEISO_PAY_L402=0                          # optional: hide L402 from discovery
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
# → token once; funding.payment_methods + funding.ark_address / funding.l402
#    (live Ark/L402 not wired — use faucet when enabled)
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

## Opt-in L402 settlement

**L402** (Lightning HTTP 402) is an additional advertised payment method for the same marketplace sessions and (later) per-request gating.

> **Status: not functional currently — do not use.** Challenge minting, Lightning invoice issuance, and preimage verification are **not implemented**. Session `funding.l402` is a placeholder describing the intended wire shape. Prefer faucet for smoke tests.

| Variable | Role |
|----------|------|
| `SEISO_PAY_L402` | Default `1` — advertise L402 in discovery/funding; set `0` to hide |
| (future) Lightning node / custodian | Issue BOLT-11 invoices + verify preimages |
| (future) macaroon root key | Mint/verify L402 macaroons scoped to session or endpoint |

Intended client flow (when wired):

1. Buyer hits pay sidecar → `402` / funding block with macaroon + invoice.
2. Pay the BOLT-11 invoice in any Lightning wallet; capture the preimage.
3. Retry with `Authorization: L402 <macaroon>:<preimage>` (or exchange for a `seiso_pay_*` session token).

Reference: [L402 payments explained](https://lightningfaucet.com/learn/l402-payments-explained/) (Lightning Faucet).

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

Never post `SEISO_PAY_TOKEN`, nsecs, HF tokens, Ark private material, L402 macaroons/preimages, or Lightning node credentials to the channel.

## Security notes (operators)

- Forge is **single-tenant**; do not share one Forge UI as multiuser SaaS.
- Session create on the sidecar is open to anyone who can reach it — put TLS + rate limits in front; keep faucet off in public.
- Prefer fixed presets over buyer-supplied config paths until allowlisted.
- Details: [docs/README.md](../README.md) · multiuser posture discussed in nest research notes.

See also: [training/mesh.md](../training/mesh.md) (experimental Buzz shared training, no fee).
