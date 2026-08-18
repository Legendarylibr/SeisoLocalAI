// SPDX-License-Identifier: GPL-3.0-or-later
pragma solidity ^0.8.24;

import {Test} from "forge-std/Test.sol";
import {SeisoPriceOracle} from "../src/SeisoPriceOracle.sol";
import {SeisoPayRouter} from "../src/SeisoPayRouter.sol";
import {IERC20} from "../src/interfaces/IERC20.sol";

contract MockUSDC is IERC20 {
    string public name = "USD Coin";
    mapping(address => uint256) public override balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;

    function decimals() external pure returns (uint8) {
        return 6;
    }

    function mint(address to, uint256 amt) external {
        balanceOf[to] += amt;
    }

    function transfer(address to, uint256 amount) external returns (bool) {
        balanceOf[msg.sender] -= amount;
        balanceOf[to] += amount;
        return true;
    }

    function transferFrom(address from, address to, uint256 amount) external returns (bool) {
        uint256 a = allowance[from][msg.sender];
        require(a >= amount, "allow");
        allowance[from][msg.sender] = a - amount;
        balanceOf[from] -= amount;
        balanceOf[to] += amount;
        return true;
    }

    function approve(address spender, uint256 amount) external returns (bool) {
        allowance[msg.sender][spender] = amount;
        return true;
    }
}

contract SeisoPayTest is Test {
    SeisoPriceOracle oracle;
    SeisoPayRouter router;
    MockUSDC usdc;

    uint256 operatorPk = 0xA11CE;
    address operator;
    address treasury = address(0x2222);
    address buyer = address(0xB0B0);

    function setUp() public {
        operator = vm.addr(operatorPk);
        oracle = new SeisoPriceOracle(address(this), address(this));
        // $100,000 / BTC and $2,500 / ETH (8 decimals)
        oracle.setFallbackPrice(oracle.BTC_USD(), 100_000 * 1e8);
        oracle.setFallbackPrice(oracle.ETH_USD(), 2_500 * 1e8);
        usdc = new MockUSDC();
        router = new SeisoPayRouter(address(this), operator, treasury, 500, usdc, oracle);
        vm.deal(buyer, 100 ether);
        usdc.mint(buyer, 1_000_000e6);
        vm.prank(buyer);
        usdc.approve(address(router), type(uint256).max);
    }

    function _sign(SeisoPayRouter.Quote memory q) internal view returns (bytes memory) {
        bytes32 digest = router.quoteHash(q);
        (uint8 v, bytes32 r, bytes32 s) = vm.sign(operatorPk, digest);
        return abi.encodePacked(r, s, v);
    }

    function test_feeCeil() public view {
        // 10 sats * 500 bps = 0.5 → ceil 1
        assertEq(router.protocolFeeSats(10, 500), 1);
        assertEq(router.totalSats(10_000, 500), 10_500);
    }

    function test_requiredUsdc() public view {
        // 10_500 sats * 100_000e8 / 1e10 = 10_500 * 1e5 / 1e2 wait:
        // 10500 * 100000e8 / 1e10 = 10500 * 1e13 / 1e10 = 10500 * 1e3 = 10_500_000
        // $10.50 USDC (6 decimals)
        assertEq(router.requiredUsdc(10_500), 10_500_000);
    }

    function test_requiredWei() public view {
        // 10500 * 100000e8 * 1e10 / 2500e8
        // = 10500 * 1e13 * 1e10 / 2.5e11
        // = 10500 * 1e23 / 2.5e11
        // = 10500 * 4e11 = 4.2e15
        assertEq(router.requiredWei(10_500), 4.2e15);
    }

    function test_payETH_splits_and_refunds() public {
        SeisoPayRouter.Quote memory q = SeisoPayRouter.Quote({
            requestId: keccak256("req-1"),
            computeSats: 10_000,
            feeBps: 500,
            deadline: block.timestamp + 600,
            asset: router.ASSET_ETH(),
            payer: address(0)
        });
        bytes memory sig = _sign(q);
        uint256 need = router.requiredWei(10_500);
        uint256 treaBefore = treasury.balance;

        vm.prank(buyer);
        router.payETH{value: need + 1 ether}(q, sig);

        assertTrue(router.spent(q.requestId));
        assertEq(treasury.balance - treaBefore, router.requiredWei(10_500) * 500 / 10_500);
        // buyer refunded extra 1 ether
        assertGt(buyer.balance, 100 ether - need - 0.01 ether);
    }

    function test_payETH_replay() public {
        SeisoPayRouter.Quote memory q = SeisoPayRouter.Quote({
            requestId: keccak256("req-2"),
            computeSats: 10_000,
            feeBps: 500,
            deadline: block.timestamp + 600,
            asset: router.ASSET_ETH(),
            payer: address(0)
        });
        bytes memory sig = _sign(q);
        uint256 need = router.requiredWei(10_500);
        vm.prank(buyer);
        router.payETH{value: need}(q, sig);
        vm.prank(buyer);
        vm.expectRevert(SeisoPayRouter.AlreadySpent.selector);
        router.payETH{value: need}(q, sig);
    }

    function test_payUSDC() public {
        SeisoPayRouter.Quote memory q = SeisoPayRouter.Quote({
            requestId: keccak256("req-3"),
            computeSats: 10_000,
            feeBps: 500,
            deadline: block.timestamp + 600,
            asset: router.ASSET_USDC(),
            payer: buyer
        });
        bytes memory sig = _sign(q);
        uint256 need = router.requiredUsdc(10_500);
        vm.prank(buyer);
        router.payUSDC(q, sig);
        assertEq(usdc.balanceOf(operator) + usdc.balanceOf(treasury), need);
        assertTrue(router.spent(q.requestId));
    }

    function test_staleFallbackReverts() public {
        vm.warp(block.timestamp + 2 hours);
        vm.expectRevert(SeisoPriceOracle.StalePrice.selector);
        oracle.ethUsd();
    }

    function test_oracleCanRefreshPrice() public {
        vm.warp(block.timestamp + 2 hours);
        oracle.setFallbackPrice(oracle.ETH_USD(), 3_000 * 1e8);
        (uint256 p,) = oracle.ethUsd();
        assertEq(p, 3_000 * 1e8);
    }
}
