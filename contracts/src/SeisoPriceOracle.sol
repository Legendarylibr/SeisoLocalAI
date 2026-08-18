// SPDX-License-Identifier: GPL-3.0-or-later
pragma solidity ^0.8.24;

import {AggregatorV3Interface} from "./interfaces/AggregatorV3Interface.sol";

/// @title SeisoPriceOracle
/// @notice ETH/USD and BTC/USD with 8 decimals (Chainlink convention).
///         Prefers a live Chainlink feed when fresh; otherwise a signed
///         fallback set by the oracle role. Stale prices revert.
/// @dev Pair ids: keccak256("ETH/USD"), keccak256("BTC/USD").
contract SeisoPriceOracle {
    bytes32 public constant ETH_USD = keccak256("ETH/USD");
    bytes32 public constant BTC_USD = keccak256("BTC/USD");

    uint256 public constant PRICE_DECIMALS = 8;

    address public owner;
    address public oracle;

    uint256 public maxStaleness = 1 hours;

    mapping(bytes32 => AggregatorV3Interface) public feeds;
    mapping(bytes32 => uint256) public fallbackPrice;
    mapping(bytes32 => uint256) public fallbackUpdatedAt;

    event OwnerTransferred(address indexed previous, address indexed current);
    event OracleUpdated(address indexed previous, address indexed current);
    event FeedUpdated(bytes32 indexed pair, address feed);
    event FallbackPriceSet(bytes32 indexed pair, uint256 price, uint256 updatedAt);
    event MaxStalenessUpdated(uint256 seconds_);

    error NotOwner();
    error NotOracle();
    error BadPrice();
    error StalePrice();
    error NoPrice();
    error ZeroAddress();

    modifier onlyOwner() {
        if (msg.sender != owner) revert NotOwner();
        _;
    }

    modifier onlyOracle() {
        if (msg.sender != oracle && msg.sender != owner) revert NotOracle();
        _;
    }

    constructor(address owner_, address oracle_) {
        if (owner_ == address(0) || oracle_ == address(0)) revert ZeroAddress();
        owner = owner_;
        oracle = oracle_;
    }

    function transferOwner(address next) external onlyOwner {
        if (next == address(0)) revert ZeroAddress();
        emit OwnerTransferred(owner, next);
        owner = next;
    }

    function setOracle(address next) external onlyOwner {
        if (next == address(0)) revert ZeroAddress();
        emit OracleUpdated(oracle, next);
        oracle = next;
    }

    function setFeed(bytes32 pair, AggregatorV3Interface feed) external onlyOwner {
        feeds[pair] = feed;
        emit FeedUpdated(pair, address(feed));
    }

    function setMaxStaleness(uint256 seconds_) external onlyOwner {
        if (seconds_ < 30 || seconds_ > 2 days) revert BadPrice();
        maxStaleness = seconds_;
        emit MaxStalenessUpdated(seconds_);
    }

    /// @notice Push a fallback price (8 decimals). Used on testnets or if
    ///         Chainlink is unset. Oracle role only.
    function setFallbackPrice(bytes32 pair, uint256 price) external onlyOracle {
        if (price == 0) revert BadPrice();
        fallbackPrice[pair] = price;
        fallbackUpdatedAt[pair] = block.timestamp;
        emit FallbackPriceSet(pair, price, block.timestamp);
    }

    /// @return price USD per 1 whole unit (ETH or BTC), 8 decimals
    /// @return updatedAt unix time of the winning source
    function getPrice(bytes32 pair) public view returns (uint256 price, uint256 updatedAt) {
        AggregatorV3Interface feed = feeds[pair];
        if (address(feed) != address(0)) {
            (
                uint80 roundId,
                int256 answer,
                ,
                uint256 feedUpdatedAt,
                uint80 answeredInRound
            ) = feed.latestRoundData();
            if (
                answer > 0 && answeredInRound >= roundId && feedUpdatedAt != 0
                    && block.timestamp - feedUpdatedAt <= maxStaleness
            ) {
                uint8 dec = feed.decimals();
                uint256 raw = uint256(answer);
                if (dec == PRICE_DECIMALS) {
                    return (raw, feedUpdatedAt);
                }
                if (dec < PRICE_DECIMALS) {
                    return (raw * (10 ** (PRICE_DECIMALS - dec)), feedUpdatedAt);
                }
                return (raw / (10 ** (dec - PRICE_DECIMALS)), feedUpdatedAt);
            }
        }

        uint256 fb = fallbackPrice[pair];
        uint256 fbAt = fallbackUpdatedAt[pair];
        if (fb == 0 || fbAt == 0) revert NoPrice();
        if (block.timestamp - fbAt > maxStaleness) revert StalePrice();
        return (fb, fbAt);
    }

    function ethUsd() external view returns (uint256 price, uint256 updatedAt) {
        return getPrice(ETH_USD);
    }

    function btcUsd() external view returns (uint256 price, uint256 updatedAt) {
        return getPrice(BTC_USD);
    }
}
