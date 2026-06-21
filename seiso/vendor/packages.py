"""Known third-party vendor trees bundled with Seiso."""

from __future__ import annotations

from seiso.vendor.bootstrap import VendorBootstrap, make_vendor_bootstrap

CODELLAMA: VendorBootstrap = make_vendor_bootstrap(
    "codellama-compress",
    "codellama_compress",
    missing_hint="Expected third_party/codellama-compress/src/codellama_compress",
)
ADAPTIVE_QUANT: VendorBootstrap = make_vendor_bootstrap(
    "adaptive-rl-quant",
    "adaptive_quant",
    missing_hint="Expected third_party/adaptive-rl-quant/src/adaptive_quant",
)
