// SPDX-License-Identifier: GPL-3.0-only
pragma solidity ^0.8.26;

import {IERC20} from "./interfaces/IERC20.sol";
import {AggregatorV3Interface} from "./interfaces/AggregatorV3Interface.sol";

/// @title SeisoPayRouter — per-request EVM payment router for x402 marketplace.
/// @notice Accepts ETH and USDC payments for Seiso marketplace requests.
///         Emits RequestPaid on successful payment. Compatible with x402
///         PAYMENT-REQUIRED / PAYMENT-SIGNATURE handshake across all EVM chains.
/// @dev Deploy once per chain. Supports ETH native + any ERC-20 (USDC, USDT, etc.).
contract SeisoPayRouter {
    error PaymentTooLow(uint256 required, uint256 received);
    error QuoteExpired(uint256 deadline, uint256 now);
    error RouterPaused();

    event RequestPaid(
        bytes32 indexed requestId,
        address indexed payer,
        address indexed payee,
        address asset,
        uint256 amount,
        uint256 protocolFee
    );

    struct Quote {
        bytes32 requestId;
        address payee;
        address asset;       // address(0) for ETH native
        uint256 amount;       // USDC atomic (6 dec) or wei
        uint256 protocolFeeBps;
        uint256 deadline;
    }

    address public immutable protocolTreasury;
    uint256 public protocolFeeBps;
    bool public paused;
    AggregatorV3Interface public priceOracle;

    constructor(address _protocolTreasury, uint256 _protocolFeeBps, address _oracle) {
        protocolTreasury = _protocolTreasury;
        protocolFeeBps = _protocolFeeBps;
        priceOracle = AggregatorV3Interface(_oracle);
    }

    modifier notPaused() {
        if (paused) revert RouterPaused();
        _;
    }

    /// @notice Pay for a request with ETH native.
    /// @param quote Signed quote from the Seiso operator.
    /// @param clientData Arbitrary client data (pass through).
    function payETH(Quote calldata quote, bytes calldata clientData)
        external
        payable
        notPaused
        returns (bool)
    {
        if (block.timestamp > quote.deadline) revert QuoteExpired(quote.deadline, block.timestamp);
        if (msg.value < quote.amount) revert PaymentTooLow(quote.amount, msg.value);

        uint256 fee = (quote.amount * quote.protocolFeeBps) / 10_000;
        uint256 net = quote.amount - fee;

        (bool okPayee, ) = quote.payee.call{value: net}("");
        require(okPayee, "payee transfer failed");

        if (fee > 0 && protocolTreasury != address(0)) {
            (bool okFee, ) = protocolTreasury.call{value: fee}("");
            require(okFee, "protocol fee transfer failed");
        }

        uint256 refund = msg.value - quote.amount;
        if (refund > 0) {
            (bool okRefund, ) = msg.sender.call{value: refund}("");
            require(okRefund, "refund failed");
        }

        emit RequestPaid(
            quote.requestId, msg.sender, quote.payee, address(0), quote.amount, fee
        );
        return true;
    }

    /// @notice Pay for a request with an ERC-20 token (USDC, USDT, etc.).
    /// @param quote Signed quote from the operator.
    /// @param clientData Arbitrary client data.
    function payERC20(Quote calldata quote, bytes calldata clientData)
        external
        notPaused
        returns (bool)
    {
        if (block.timestamp > quote.deadline) revert QuoteExpired(quote.deadline, block.timestamp);
        if (quote.asset == address(0)) revert("ERC20: asset required");

        uint256 fee = (quote.amount * quote.protocolFeeBps) / 10_000;
        uint256 net = quote.amount - fee;

        IERC20 token = IERC20(quote.asset);

        require(token.transferFrom(msg.sender, quote.payee, net), "payee transfer failed");
        if (fee > 0 && protocolTreasury != address(0)) {
            require(token.transferFrom(msg.sender, protocolTreasury, fee), "protocol fee failed");
        }

        emit RequestPaid(
            quote.requestId, msg.sender, quote.payee, quote.asset, quote.amount, fee
        );
        return true;
    }

    /// @notice Set protocol fee (owner / operator).
    function setProtocolFeeBps(uint256 _bps) external {
        require(msg.sender == protocolTreasury || msg.sender == address(this), "unauthorized");
        require(_bps <= 1000, "max 10%");
        protocolFeeBps = _bps;
    }

    /// @notice Toggle pause for maintenance.
    function setPaused(bool _paused) external {
        require(msg.sender == protocolTreasury, "unauthorized");
        paused = _paused;
    }

    /// @notice Set price oracle address.
    function setOracle(address _oracle) external {
        require(msg.sender == protocolTreasury, "unauthorized");
        priceOracle = AggregatorV3Interface(_oracle);
    }

    /// @notice Get latest ETH/USD price from oracle (8 decimals).
    function getEthUsdPrice() external view returns (uint256, uint8) {
        (, int256 price,,,) = priceOracle.latestRoundData();
        return (uint256(price), 8);
    }
}
