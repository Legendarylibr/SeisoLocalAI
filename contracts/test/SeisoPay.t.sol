// SPDX-License-Identifier: GPL-3.0-only
pragma solidity ^0.8.26;

import {Test, console} from "forge-std/Test.sol";
import {SeisoPayRouter} from "../src/SeisoPayRouter.sol";
import {SeisoPriceOracle} from "../src/SeisoPriceOracle.sol";
import {IERC20} from "../src/interfaces/IERC20.sol";

contract SeisoPayRouterTest is Test {
    SeisoPayRouter public router;
    SeisoPriceOracle public oracle;
    address public treasury;
    address public operator;
    address public buyer;
    address public mockUsdc;

    bytes32 constant REQUEST_ID = keccak256("test-request-1");
    uint256 constant DEADLINE = 1_000_000_000;
    uint256 constant AMOUNT = 100_000; // 0.1 USDC (6 decimals)
    uint256 constant FEE_BPS = 500;   // 5%

    function setUp() public {
        treasury = makeAddr("treasury");
        operator = makeAddr("operator");
        buyer = makeAddr("buyer");
        mockUsdc = makeAddr("USDC");

        // Deploy with mock oracle address
        oracle = new SeisoPriceOracle(address(0), address(0));
        router = new SeisoPayRouter(treasury, FEE_BPS, address(oracle));
    }

    function test_Deployment() public view {
        assertEq(router.protocolTreasury(), treasury);
        assertEq(router.protocolFeeBps(), FEE_BPS);
        assertEq(address(router.priceOracle()), address(oracle));
        assertFalse(router.paused());
    }

    function test_PayETH() public {
        SeisoPayRouter.Quote memory q = SeisoPayRouter.Quote({
            requestId: REQUEST_ID,
            payee: operator,
            asset: address(0),
            amount: AMOUNT,
            protocolFeeBps: FEE_BPS,
            deadline: DEADLINE
        });

        vm.deal(buyer, AMOUNT * 2);
        vm.prank(buyer);

        uint256 fee = (AMOUNT * FEE_BPS) / 10_000;
        uint256 net = AMOUNT - fee;
        uint256 operatorBefore = operator.balance;
        uint256 treasuryBefore = treasury.balance;

        vm.expectEmit(true, true, true, true);
        emit SeisoPayRouter.RequestPaid(REQUEST_ID, buyer, operator, address(0), AMOUNT, fee);
        router.payETH{value: AMOUNT}(q, bytes("test"));

        assertEq(operator.balance - operatorBefore, net);
        assertEq(treasury.balance - treasuryBefore, fee);
    }

    function test_RevertIfExpired() public {
        SeisoPayRouter.Quote memory q = SeisoPayRouter.Quote({
            requestId: REQUEST_ID,
            payee: operator,
            asset: address(0),
            amount: AMOUNT,
            protocolFeeBps: FEE_BPS,
            deadline: 1 // already expired
        });

        vm.deal(buyer, AMOUNT);
        vm.prank(buyer);
        vm.expectRevert(abi.encodeWithSelector(SeisoPayRouter.QuoteExpired.selector, 1, block.timestamp));
        router.payETH{value: AMOUNT}(q, bytes("test"));
    }

    function test_RevertIfPaused() public {
        vm.prank(treasury);
        router.setPaused(true);

        SeisoPayRouter.Quote memory q = SeisoPayRouter.Quote({
            requestId: REQUEST_ID,
            payee: operator,
            asset: address(0),
            amount: AMOUNT,
            protocolFeeBps: FEE_BPS,
            deadline: DEADLINE
        });

        vm.deal(buyer, AMOUNT);
        vm.prank(buyer);
        vm.expectRevert(SeisoPayRouter.RouterPaused.selector);
        router.payETH{value: AMOUNT}(q, bytes("test"));
    }

    function test_SetProtocolFeeBps() public {
        vm.prank(treasury);
        router.setProtocolFeeBps(100); // 1%
        assertEq(router.protocolFeeBps(), 100);
    }

    function test_RevertFeeTooHigh() public {
        vm.prank(treasury);
        vm.expectRevert();
        router.setProtocolFeeBps(2000); // > 10%
    }
}

contract SeisoPriceOracleTest is Test {
    SeisoPriceOracle public oracle;

    function setUp() public {
        oracle = new SeisoPriceOracle(address(0), address(0));
    }

    function test_Deployment() public view {
        assertEq(address(oracle.btcUsd()), address(0));
        assertEq(address(oracle.ethUsd()), address(0));
    }
}
