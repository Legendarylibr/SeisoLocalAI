// SPDX-License-Identifier: GPL-3.0-or-later
pragma solidity ^0.8.24;

import {IERC20} from "./interfaces/IERC20.sol";
import {SeisoPriceOracle} from "./SeisoPriceOracle.sol";

/// @title SeisoPayRouter
/// @notice Per-request marketplace settlement: buyer pays **this request** in
///         USDC or ETH. Protocol fee is taken on top (ceil bps) and split
///         trustlessly to operator + treasury. Prices come from
///         {SeisoPriceOracle} (Chainlink + updateable fallback).
///
/// Quote is operator-signed (EIP-712) so the buyer knows exact sats and the
/// contract cannot be griefed with a different amount. `requestId` is
/// single-use.
///
/// Math (must match `seiso.pay.fx`):
///   usdc_6 = ceil(sats * btcUsd_8 / 1e10)
///   wei    = ceil(sats * btcUsd_8 * 1e10 / ethUsd_8)
contract SeisoPayRouter {
    uint16 public constant MAX_FEE_BPS = 1000; // 10%
    uint16 public constant BPS_DENOM = 10_000;
    uint256 public constant SATS_PER_BTC = 100_000_000;
    uint256 public constant PRICE_DECIMALS = 8;

    bytes32 public constant QUOTE_TYPEHASH = keccak256(
        "Quote(bytes32 requestId,uint256 computeSats,uint16 feeBps,uint256 deadline,uint8 asset,address payer)"
    );

    /// @dev 0 = ETH, 1 = USDC
    uint8 public constant ASSET_ETH = 0;
    uint8 public constant ASSET_USDC = 1;

    address public owner;
    address public operator;
    address public treasury;
    uint16 public feeBps;
    IERC20 public usdc;
    SeisoPriceOracle public oracle;

    mapping(bytes32 => bool) public spent;

    bool private _locked;

    struct Quote {
        bytes32 requestId;
        uint256 computeSats;
        uint16 feeBps;
        uint256 deadline;
        uint8 asset;
        address payer; // address(0) = any
    }

    event OwnerTransferred(address indexed previous, address indexed current);
    event OperatorUpdated(address indexed operator);
    event TreasuryUpdated(address indexed treasury);
    event FeeBpsUpdated(uint16 feeBps);
    event OracleUpdated(address indexed oracle);
    event UsdcUpdated(address indexed usdc);
    event RequestPaid(
        bytes32 indexed requestId,
        address indexed payer,
        uint8 asset,
        uint256 computeSats,
        uint256 protocolFeeSats,
        uint256 paidAtomic,
        uint256 operatorAtomic,
        uint256 treasuryAtomic
    );

    error NotOwner();
    error ZeroAddress();
    error BadFee();
    error BadQuote();
    error Expired();
    error AlreadySpent();
    error BadSignature();
    error WrongPayer();
    error WrongAsset();
    error InsufficientPayment();
    error TransferFailed();
    error Reentrant();

    modifier onlyOwner() {
        if (msg.sender != owner) revert NotOwner();
        _;
    }

    modifier nonReentrant() {
        if (_locked) revert Reentrant();
        _locked = true;
        _;
        _locked = false;
    }

    constructor(
        address owner_,
        address operator_,
        address treasury_,
        uint16 feeBps_,
        IERC20 usdc_,
        SeisoPriceOracle oracle_
    ) {
        if (
            owner_ == address(0) || operator_ == address(0) || treasury_ == address(0)
                || address(oracle_) == address(0)
        ) revert ZeroAddress();
        if (feeBps_ > MAX_FEE_BPS) revert BadFee();
        owner = owner_;
        operator = operator_;
        treasury = treasury_;
        feeBps = feeBps_;
        usdc = usdc_;
        oracle = oracle_;
    }

    function transferOwner(address next) external onlyOwner {
        if (next == address(0)) revert ZeroAddress();
        emit OwnerTransferred(owner, next);
        owner = next;
    }

    function setOperator(address next) external onlyOwner {
        if (next == address(0)) revert ZeroAddress();
        operator = next;
        emit OperatorUpdated(next);
    }

    function setTreasury(address next) external onlyOwner {
        if (next == address(0)) revert ZeroAddress();
        treasury = next;
        emit TreasuryUpdated(next);
    }

    function setFeeBps(uint16 next) external onlyOwner {
        if (next > MAX_FEE_BPS) revert BadFee();
        feeBps = next;
        emit FeeBpsUpdated(next);
    }

    function setOracle(SeisoPriceOracle next) external onlyOwner {
        if (address(next) == address(0)) revert ZeroAddress();
        oracle = next;
        emit OracleUpdated(address(next));
    }

    function setUsdc(IERC20 next) external onlyOwner {
        usdc = next;
        emit UsdcUpdated(address(next));
    }

    /// @notice ceil(compute * bps / 10000)
    function protocolFeeSats(uint256 computeSats, uint16 bps) public pure returns (uint256) {
        if (bps > MAX_FEE_BPS) revert BadFee();
        return (computeSats * uint256(bps) + (BPS_DENOM - 1)) / BPS_DENOM;
    }

    function totalSats(uint256 computeSats, uint16 bps) public pure returns (uint256) {
        return computeSats + protocolFeeSats(computeSats, bps);
    }

    /// @notice USDC atomic (6 decimals) for `sats` at current BTC/USD.
    function requiredUsdc(uint256 sats) public view returns (uint256) {
        (uint256 btcUsd,) = oracle.btcUsd();
        // sats * btcUsd_8 / 1e10  (ceil)
        return _ceilDiv(sats * btcUsd, 10 ** 10);
    }

    /// @notice wei for `sats` at current BTC/USD and ETH/USD.
    function requiredWei(uint256 sats) public view returns (uint256) {
        (uint256 btcUsd,) = oracle.btcUsd();
        (uint256 ethUsd,) = oracle.ethUsd();
        if (ethUsd == 0) revert BadQuote();
        // sats * btcUsd_8 * 1e10 / ethUsd_8  (ceil)
        return _ceilDiv(sats * btcUsd * (10 ** 10), ethUsd);
    }

    function payETH(Quote calldata q, bytes calldata sig) external payable nonReentrant {
        _checkQuote(q, sig, ASSET_ETH);
        uint256 dueSats = totalSats(q.computeSats, q.feeBps);
        uint256 need = requiredWei(dueSats);
        if (msg.value < need) revert InsufficientPayment();

        uint256 feeSats = protocolFeeSats(q.computeSats, q.feeBps);
        uint256 trea = _ceilDiv(need * feeSats, dueSats);
        uint256 op = need - trea;

        spent[q.requestId] = true;

        _sendEth(operator, op);
        _sendEth(treasury, trea);
        uint256 refund = msg.value - need;
        if (refund > 0) _sendEth(msg.sender, refund);

        emit RequestPaid(
            q.requestId, msg.sender, ASSET_ETH, q.computeSats, feeSats, need, op, trea
        );
    }

    function payUSDC(Quote calldata q, bytes calldata sig) external nonReentrant {
        _checkQuote(q, sig, ASSET_USDC);
        if (address(usdc) == address(0)) revert ZeroAddress();
        uint256 dueSats = totalSats(q.computeSats, q.feeBps);
        uint256 need = requiredUsdc(dueSats);

        uint256 feeSats = protocolFeeSats(q.computeSats, q.feeBps);
        uint256 trea = _ceilDiv(need * feeSats, dueSats);
        uint256 op = need - trea;

        spent[q.requestId] = true;

        if (!usdc.transferFrom(msg.sender, address(this), need)) revert TransferFailed();
        if (op > 0 && !usdc.transfer(operator, op)) revert TransferFailed();
        if (trea > 0 && !usdc.transfer(treasury, trea)) revert TransferFailed();

        emit RequestPaid(
            q.requestId, msg.sender, ASSET_USDC, q.computeSats, feeSats, need, op, trea
        );
    }

    function domainSeparator() public view returns (bytes32) {
        return keccak256(
            abi.encode(
                keccak256(
                    "EIP712Domain(string name,string version,uint256 chainId,address verifyingContract)"
                ),
                keccak256(bytes("SeisoPayRouter")),
                keccak256(bytes("1")),
                block.chainid,
                address(this)
            )
        );
    }

    function quoteHash(Quote calldata q) public view returns (bytes32) {
        return keccak256(
            abi.encodePacked(
                "\x19\x01",
                domainSeparator(),
                keccak256(
                    abi.encode(
                        QUOTE_TYPEHASH,
                        q.requestId,
                        q.computeSats,
                        q.feeBps,
                        q.deadline,
                        q.asset,
                        q.payer
                    )
                )
            )
        );
    }

    function _checkQuote(Quote calldata q, bytes calldata sig, uint8 expectedAsset) internal {
        if (spent[q.requestId]) revert AlreadySpent();
        if (q.deadline < block.timestamp) revert Expired();
        if (q.feeBps != feeBps) revert BadFee();
        if (q.asset != expectedAsset) revert WrongAsset();
        if (q.payer != address(0) && q.payer != msg.sender) revert WrongPayer();
        if (q.computeSats == 0) revert BadQuote();

        bytes32 digest = quoteHash(q);
        address recovered = _recover(digest, sig);
        if (recovered != operator) revert BadSignature();
    }

    function _recover(bytes32 digest, bytes calldata sig) internal pure returns (address) {
        if (sig.length != 65) revert BadSignature();
        bytes32 r;
        bytes32 s;
        uint8 v;
        assembly ("memory-safe") {
            r := calldataload(sig.offset)
            s := calldataload(add(sig.offset, 32))
            v := byte(0, calldataload(add(sig.offset, 64)))
        }
        if (v < 27) v += 27;
        if (v != 27 && v != 28) revert BadSignature();
        address recovered = ecrecover(digest, v, r, s);
        if (recovered == address(0)) revert BadSignature();
        return recovered;
    }

    function _ceilDiv(uint256 a, uint256 b) internal pure returns (uint256) {
        if (b == 0) revert BadQuote();
        return (a + b - 1) / b;
    }

    function _sendEth(address to, uint256 amount) internal {
        if (amount == 0) return;
        (bool ok,) = to.call{value: amount}("");
        if (!ok) revert TransferFailed();
    }
}
