// SPDX-License-Identifier: GPL-3.0-only
pragma solidity ^0.8.26;

import {AggregatorV3Interface} from "./interfaces/AggregatorV3Interface.sol";

/// @title SeisoPriceOracle — on-chain price feed for Seiso marketplace FX.
/// @notice Wraps Chainlink AggregatorV3Interface for BTC/USD and ETH/USD.
///         Deploy once per chain. Supports any Chainlink feed address.
/// @dev The operator's pay sidecar reads these prices to quote sats → USDC/ETH.
contract SeisoPriceOracle {
    AggregatorV3Interface public btcUsd;
    AggregatorV3Interface public ethUsd;
    address public immutable owner;

    event PricesUpdated(uint256 btcUsd8, uint256 ethUsd8, uint256 timestamp);
    event OracleAddressesUpdated(address btc, address eth);

    error StalePrice(uint256 updatedAt, uint256 maxAge);

    constructor(address _btcUsd, address _ethUsd) {
        owner = msg.sender;
        btcUsd = AggregatorV3Interface(_btcUsd);
        ethUsd = AggregatorV3Interface(_ethUsd);
    }

    /// @notice Read latest BTC/USD and ETH/USD prices (8 decimals).
    /// @param maxAge Max age in seconds before price is considered stale.
    function latestPrices(uint256 maxAge)
        external
        view
        returns (uint256 btcUsd8, uint256 ethUsd8, uint256 timestamp)
    {
        (, int256 btcPrice,, uint256 btcTime,) = btcUsd.latestRoundData();
        (, int256 ethPrice,, uint256 ethTime,) = ethUsd.latestRoundData();

        if (maxAge > 0) {
            if (block.timestamp - btcTime > maxAge) revert StalePrice(btcTime, maxAge);
            if (block.timestamp - ethTime > maxAge) revert StalePrice(ethTime, maxAge);
        }

        return (uint256(btcPrice), uint256(ethPrice), block.timestamp);
    }

    /// @notice Update oracle contract addresses (owner only).
    function setOracles(address _btcUsd, address _ethUsd) external {
        require(msg.sender == owner, "unauthorized");
        btcUsd = AggregatorV3Interface(_btcUsd);
        ethUsd = AggregatorV3Interface(_ethUsd);
        emit OracleAddressesUpdated(_btcUsd, _ethUsd);
    }
}
