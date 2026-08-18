# Seiso marketplace contracts

Per-request settlement on EVM. **Audit before mainnet.** GPL-3.0-or-later.

| Contract | Role |
|----------|------|
| `SeisoPriceOracle` | ETH/USD + BTC/USD (8 decimals). Prefers [Chainlink](https://docs.chain.link/data-feeds/price-feeds/addresses) `AggregatorV3Interface` when the round is fresh; otherwise an oracle-role fallback (`setFallbackPrice`). Stale prices revert. |
| `SeisoPayRouter` | One `requestId` → one payment in **ETH** or **USDC**. Operator EIP-712 quote. Protocol fee (ceil bps, default 5%) split on-chain to operator + treasury. Surplus ETH refunded. |

## Pricing math

Quotes stay sat-denominated (same as Lightning / Ark). Conversion uses the oracle:

```
usdc_6 = ceil(sats * btcUsd_8 / 1e10)
wei    = ceil(sats * btcUsd_8 * 1e10 / ethUsd_8)
```

Python `seiso.pay.fx` is bit-identical. Updating the ETH or BTC fallback (or rotating the Chainlink feed) immediately changes `requiredWei` / `requiredUsdc`.

## Deploy (Foundry)

```bash
cd contracts
forge install foundry-rs/forge-std --no-git   # once
forge test
forge script script/Deploy.s.sol --rpc-url $RPC --broadcast
```

Constructor args: owner, operator (signer), treasury, feeBps (≤ 1000), USDC, oracle.

Bind feeds after deploy:

```solidity
oracle.setFeed(oracle.ETH_USD(), AggregatorV3Interface(CHAINLINK_ETH_USD));
oracle.setFeed(oracle.BTC_USD(), AggregatorV3Interface(CHAINLINK_BTC_USD));
```

Known Chainlink ETH/USD (verify before use):

| Network | Feed |
|---------|------|
| Ethereum | `0x5f4eC3Df9cbd43714FE2740f5E3616155c5b8419` |
| Base | `0x71041dddad3595F9CEd3DcCFBe3D1F4b0a16Bb70` |

## Buyer flow

1. `POST /pay/v1/requests` or unpaid `POST /v1/chat/completions` → HTTP 402 + `PAYMENT-REQUIRED`.
2. Sign the operator `Quote` and call `payETH` or `payUSDC`.
3. Retry chat with `X-Seiso-Request-Id`.

Live facilitator / wallet wiring is operator-side. Sim: `SEISO_PAY_X402_SIM=1`.
