# Per-request marketplace payments

> Live rails still need an operator-run Chainlink/oracle + `SeisoPayRouter` deploy (or L402/Ark client). Sim: `SEISO_PAY_X402_SIM=1` / faucet. Do not use real funds until those are wired and reviewed.

Self-hosted Forge stays **free**. This page is only for the opt-in pay sidecar.

## What is deducted

Each inference HTTP request has its own quote (`compute_sats` + 5% protocol fee). The buyer pays **that request**, not a prepaid blob that is later skimmed.

| Rail | Unit | How |
|------|------|-----|
| **x402 USDC** | USDC atomic (6 dec) | HTTP 402 + `PAYMENT-REQUIRED` / `PAYMENT-SIGNATURE` ([x402.org](https://x402.org/)) |
| **ETH** | wei | Same quote converted with `SeisoPriceOracle` ETH/USD + BTC/USD; `SeisoPayRouter.payETH` |
| **L402** | sats | Lightning HTTP 402 invoice **per request** |
| **Ark / BTC HTLC** | sats | Ark address + P2WSH HTLC / miniscript voucher for the quoted sats |

Prepaid `seiso_pay_*` sessions still work if the client sends a funded Bearer token.

## ETH / USDC price

Sats stay the list currency. Conversion (8-decimal USD prices, Chainlink-style):

```
usdc = ceil(sats * btc_usd_8 / 1e10)
wei  = ceil(sats * btc_usd_8 * 1e10 / eth_usd_8)
```

Update the market price:

- On-chain: `SeisoPriceOracle.setFallbackPrice(ETH_USD, price)` (oracle role) or bind a Chainlink feed with `setFeed`.
- Off-chain sidecar: `SEISO_PAY_ETH_USD_8` / `SEISO_PAY_BTC_USD_8` (or `SEISO_PAY_ORACLE_URL` on an allowlisted host). Stale prices fail closed.

Python: `seiso.pay.fx.quote_fx`. Solidity: `SeisoPayRouter.requiredWei` / `requiredUsdc`. Tests lock both to the same vectors.

## HTTP

```bash
# 1. Quote this call (or POST /v1/chat/completions with no Bearer)
curl -D- -X POST "$PAY/pay/v1/requests" \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":64}'
# → 402 + PAYMENT-REQUIRED + fx.wei + fx.usdc_atomic + rails

# 2. Pay (sim)
curl -X POST "$PAY/pay/v1/requests/$REQUEST_ID/complete" \
  -H 'Content-Type: application/json' \
  -d '{"via":"eth"}'          # or x402 | l402 | ark

# 3. Retry the model call
curl -X POST "$PAY/v1/chat/completions" \
  -H "X-Seiso-Request-Id: $REQUEST_ID" \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"hi"}]}'
```

## Contracts and scripts

- EVM: [`contracts/`](../../contracts/README.md) — `SeisoPriceOracle`, `SeisoPayRouter`
- BTC HTLC: `seiso.pay.btc.htlc.build_htlc`
- Ark descriptor: `seiso.pay.btc.ark_voucher.build_voucher`

There is **no Seiso token**.
