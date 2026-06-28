"""Known bundled source packages shipped with Seiso."""

from __future__ import annotations

from seiso.vendor.bootstrap import VendorBootstrap, make_vendor_bootstrap

CODELLAMA: VendorBootstrap = make_vendor_bootstrap(
    "seiso/codellama_compress",
    "seiso.codellama_compress",
    missing_hint="Expected seiso.codellama_compress",
)
ADAPTIVE_QUANT: VendorBootstrap = make_vendor_bootstrap(
    "seiso/adaptive_quant",
    "seiso.adaptive_quant",
    missing_hint="Expected seiso.adaptive_quant",
)
