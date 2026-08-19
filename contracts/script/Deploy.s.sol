// SPDX-License-Identifier: GPL-3.0-only
pragma solidity ^0.8.26;

import {Script} from "forge-std/Script.sol";
import {SeisoPayRouter} from "../src/SeisoPayRouter.sol";
import {SeisoPriceOracle} from "../src/SeisoPriceOracle.sol";

/// @notice Deploy SeisoPayRouter and SeisoPriceOracle.
/// Usage: forge script script/Deploy.s.sol --rpc-url base_sepolia --broadcast
contract DeployScript is Script {
    // Chainlink feed addresses (Base Sepolia defaults)
    address constant BTC_USD_FEED = 0x1b3b64c69c2cDbF7DEAbbB7F9A5F31e6f0e8b8D9;
    address constant ETH_USD_FEED = 0x4aDC67696bA383F43DD60A6e78d2D7b7B8e8b2F0;
    address constant PROTOCOL_TREASURY = 0x0000000000000000000000000000000000000402;

    function run() external {
        uint256 deployerKey = vm.envUint("DEPLOYER_PRIVATE_KEY");
        address deployer = vm.addr(deployerKey);
        vm.startBroadcast(deployerKey);

        // Deploy price oracle first
        SeisoPriceOracle oracle = new SeisoPriceOracle(BTC_USD_FEED, ETH_USD_FEED);

        // Deploy router with 5% protocol fee
        SeisoPayRouter router = new SeisoPayRouter(
            PROTOCOL_TREASURY,
            500, // 5%
            address(oracle)
        );

        vm.stopBroadcast();

        console.log("SeisoPriceOracle deployed at:", address(oracle));
        console.log("SeisoPayRouter deployed at:", address(router));
        console.log("Deployer:", deployer);
    }
}
