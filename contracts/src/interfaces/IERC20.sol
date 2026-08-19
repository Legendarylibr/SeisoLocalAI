// SPDX-License-Identifier: GPL-3.0-only
pragma solidity ^0.8.26;

/// @dev Minimal ERC-20 interface for SeisoPayRouter token transfers.
interface IERC20 {
    function transferFrom(address from, address to, uint256 value) external returns (bool);
    function transfer(address to, uint256 value) external returns (bool);
    function approve(address spender, uint256 value) external returns (bool);
    function balanceOf(address account) external view returns (uint256);
    function decimals() external view returns (uint8);
}
