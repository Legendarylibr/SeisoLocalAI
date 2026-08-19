# Per-request payments (x402 EVM / ETH / L402 / Ark)

Seiso supports **pay-per-call** inference alongside prepaid sessions. A client
requests a quote for N tokens, receives an **HTTP 402** with payment options
across every advertised rail, pays, and retries with a proof.

## Rails

| Rail | Asset | Network |
|------|-------|---------|
| **x402** | USDC (6-dec) | Any EVM chain (`SEISO_PAY_X402_NETWORK`) |
| **ETH** | ETH native | Same as x402 via `SeisoPayRouter.payETH` |
| **L402** | sats (BTC) | Lightning (sim) |
| **Ark** | BTC | Signet / testnet (descriptor only) |
| **BTC HTLC** | BTC | L1 P2WSH (on-chain fallback) |

## Supported EVM Chains

x402 works on every EVM chain with USDC. Configure with `SEISO_PAY_X402_NETWORK`
(CAIP-2 format):

| Chain | CAIP-2 |
|-------|--------|
| Ethereum | `eip155:1` |
| Base | `eip155:8453` |
| Arbitrum One | `eip155:42161` |
| Optimism | `eip155:10` |
| Polygon PoS | `eip155:137` |
| Avalanche C-Chain | `eip155:43114` |
| BNB Chain | `eip155:56` |
| Robinhood Chain | `eip155:4663` |
| Base Sepolia (test) | `eip155:84532` |
| Robinhood Chain Testnet | `eip155:46630` |

See `seiso.pay.x402.USDC_BY_NETWORK` for the full list (40+ networks).

## Flow

1. Client sends `POST /v1/chat/completions` with `prompt_tokens` estimate.
2. Server returns `HTTP 402` + `PAYMENT-REQUIRED` header with all rail options.
3. Client picks a rail, pays (sim or live), gets a receipt.
4. Client retries with `PAYMENT-SIGNATURE` (x402) or `Authorization: L402` or
   on-chain `SeisoPayRouter.RequestPaid` event.
5. Server serves the completion, debits the prepaid balance.

## CLI

```bash
# Quote (no charge)
seiso pay quote --type inference --prompt-tokens 500 --completion-tokens 200

# Fund via x402 (sim)
seiso pay session create --sats 5000 --x402 --network eip155:84532
seiso pay session fund --session SID --sats 5000 --x402

# Fund via x402 on Robinhood Chain
seiso pay session fund --session SID --sats 5000 --x402 --network eip155:4663
```

## Contract Deployment

Deploy `SeisoPayRouter` and `SeisoPriceOracle` on any EVM chain:

```bash
cd contracts
forge install
forge script script/Deploy.s.sol --rpc-url base_sepolia --broadcast
```

See [contracts/](../contracts/) for Solidity source and tests.
