// SPDX-License-Identifier: GPL-3.0-or-later
pragma solidity ^0.8.24;

import {Script} from "forge-std/Script.sol";
import {SeisoPriceOracle} from "../src/SeisoPriceOracle.sol";
import {SeisoPayRouter} from "../src/SeisoPayRouter.sol";
import {IERC20} from "../src/interfaces/IERC20.sol";

/// @notice Broadcast deploy. Set env: OPERATOR, TREASURY, USDC, FEE_BPS.
contract Deploy is Script {
    function run() external {
        address owner = vm.envOr("OWNER", msg.sender);
        address operator = vm.envAddress("OPERATOR");
        address treasury = vm.envAddress("TREASURY");
        address usdc = vm.envAddress("USDC");
        uint16 feeBps = uint16(vm.envOr("FEE_BPS", uint256(500)));

        vm.startBroadcast();
        SeisoPriceOracle oracle = new SeisoPriceOracle(owner, owner);
        SeisoPayRouter router =
            new SeisoPayRouter(owner, operator, treasury, feeBps, IERC20(usdc), oracle);
        vm.stopBroadcast();

        router; // silence unused in some solc paths
    }
}
