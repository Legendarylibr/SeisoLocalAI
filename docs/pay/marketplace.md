# Seiso sats marketplace (opt-in)

> **Not functional yet — do not use.** This surface is scaffolding / docs only. Do not run it for production work or real funds. Live **Ark**, **L402**, and **x402 EVM** settlement are not wired; faucet/sim only for local smoke tests.

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
| **L402** ([Lightning HTTP 402](https://lightningfaucet.com/learn/l402-payments-explained/)) | Server returns `HTTP 402` + `WWW-Authenticate: L402` with a BOLT-11 invoice and macaroon; client pays Lightning, retries with `Authorization: L402 <macaroon>:<preimage>` → session credit | **Sim ready** with `SEISO_PAY_L402_SIM=1` (or faucet); **live LN not wired** — do not use for real funds |
| **x402** ([HTTP-native EVM](https://x402.org/)) | Server returns `HTTP 402` + `PAYMENT-REQUIRED` (`exact` scheme, USDC on an EVM CAIP-2 network). Buyer retries with `PAYMENT-SIGNATURE` (EIP-3009-shaped payload) → session credit | **Sim ready** with `SEISO_PAY_X402_SIM=1` (or faucet); **live USDC / facilitator not wired** — do not use for real funds |
| **Dev faucet** | `SEISO_PAY_FAUCET=1` credits a session without chain IO | Smoke tests **only** — never on a public market |

Discovery advertises these under `payment_methods` in `GET /.well-known/seiso-pay.json` and on session `funding` payloads. Hide L402 with `SEISO_PAY_L402=0` or x402 with `SEISO_PAY_X402=0` (Ark still listed).

L402 fits agent/API buyers especially well: no accounts, machine-readable 402 challenge, sat-denominated per session or (later) per request. See the [L402 payments explained](https://lightningfaucet.com/learn/l402-payments-explained/) reference for the wire format.

## Operator (sell capacity)

```bash
export SEISO_ALLOW_PAY=1
export SEISO_PROTOCOL_TREASURY_ARK=ark1…           # required for non-faucet settles
export SEISO_OPERATOR_ARK=ark1…
export SEISO_PROTOCOL_FEE_BPS=500                  # 5%
# export SEISO_PAY_FAUCET=1                        # DEV ONLY — never on a public market
# export SEISO_PAY_L402=0                          # optional: hide L402 from discovery
# export SEISO_PAY_L402_SIM=1                      # sim L402 fund/exchange (also on with faucet)
# export SEISO_PAY_X402=0                          # optional: hide x402 from discovery
# export SEISO_PAY_X402_SIM=1                      # sim x402 EVM fund/exchange (also on with faucet)
# export SEISO_OPERATOR_EVM=0x…                    # x402 payTo (USDC receive)
# export SEISO_PAY_X402_NETWORK=eip155:84532       # Base Sepolia default
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
# Sim L402 top-up (or --faucet): 
# seiso pay session fund --session ID --sats 20000 --l402
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

**L402** (Lightning HTTP 402) funds the same prepaid marketplace sessions (not per-request auth on every job poll).

> **Live Lightning is not functional yet — do not use for real funds.** Challenge minting against a real LN node is not bundled. **Simulated** fund/exchange works with `SEISO_PAY_L402_SIM=1` (also enabled when `SEISO_PAY_FAUCET=1`).

| Variable | Role |
|----------|------|
| `SEISO_PAY_L402` | Default `1` — advertise L402 in discovery/funding; set `0` to hide |
| `SEISO_PAY_L402_SIM` | Enable simulated mint + preimage verify (credits session balance) |
| `SEISO_PAY_L402_ROOT_KEY` | Optional HMAC seed for sim macaroons |
| (future) Lightning node / custodian | Issue real BOLT-11 invoices + verify preimages |

Client flow (sim today; live LN later):

1. `POST /pay/v1/sessions/fund/l402` with `{session_id, sats}` → **HTTP 402** + `WWW-Authenticate: L402 …` + JSON challenge.
2. Pay BOLT-11 (or use `sim_preimage` in sim) and capture preimage.
3. `POST /pay/v1/sessions/fund/l402/complete` with `Authorization: L402 <macaroon>:<preimage>` → session credited; keep using `Bearer seiso_pay_*`.
4. CLI: `seiso pay session fund --session ID --sats N --l402` (auto mint+complete in sim).

Reference: [L402 payments explained](https://lightningfaucet.com/learn/l402-payments-explained/) (Lightning Faucet).

## Opt-in x402 EVM settlement

**x402** ([x402.org](https://x402.org/)) funds the same prepaid marketplace sessions using the HTTP 402 + `PAYMENT-REQUIRED` / `PAYMENT-SIGNATURE` handshake. The advertised scheme is **`exact`** on an EVM CAIP-2 network (default **Base Sepolia** `eip155:84532`, USDC).

> **Live EVM / facilitator is not functional yet — do not use for real funds.** Challenge minting against a real USDC transfer or x402 facilitator is not bundled. **Simulated** fund/exchange works with `SEISO_PAY_X402_SIM=1` (also enabled when `SEISO_PAY_FAUCET=1`).

| Variable | Role |
|----------|------|
| `SEISO_PAY_X402` | Default `1` — advertise x402 in discovery/funding; set `0` to hide |
| `SEISO_PAY_X402_SIM` | Enable simulated mint + signature verify (credits session balance) |
| `SEISO_OPERATOR_EVM` | `payTo` address (0x…) for USDC receive |
| `SEISO_PROTOCOL_TREASURY_EVM` | Optional protocol-fee EVM address (reserved) |
| `SEISO_PAY_X402_NETWORK` | CAIP-2 network (default `eip155:84532`) |
| `SEISO_PAY_X402_ASSET` | ERC-20 address override (default USDC for the network) |
| `SEISO_PAY_X402_ATOMIC_PER_SAT` | Sim mapping sats → USDC atomic (default `1`; not a market FX) |
| (future) x402 facilitator | Verify EIP-3009 `transferWithAuthorization` on-chain |

Client flow (sim today; live EVM later):

1. `POST /pay/v1/sessions/fund/x402` with `{session_id, sats}` → **HTTP 402** + `PAYMENT-REQUIRED` + `WWW-Authenticate: X402 …`.
2. Sign an `exact` payment payload (EIP-3009-shaped). In sim, use `sim_payment_signature`.
3. `POST /pay/v1/sessions/fund/x402/complete` with `PAYMENT-SIGNATURE: <payload>` → session credited; keep using `Bearer seiso_pay_*`.
4. CLI: `seiso pay session fund --session ID --sats N --x402` (auto mint+complete in sim).

Reference: [x402 seller quickstart](https://docs.x402.org/getting-started/quickstart-for-sellers).

## Job failure / cancel refunds

Jobs escrow the full quote up front. On **failure**, **cancel**, or **GPU-busy** reject, escrow is restored to the **prepaid session balance** (`refunded_sats` on the job + session; ledger `escrow_refund`).

Lightning/L402 pay-in is one-way — marketplace refunds do **not** send sats back over Lightning. Buyers reuse the restored session balance for later jobs/inference. Receipts include `refunded_sats` and `settlement.status=refunded`.

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
