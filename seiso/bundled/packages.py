"""Known bundled source packages shipped with Seiso."""

from __future__ import annotations

from seiso.bundled.bootstrap import BundledPackage, make_bundled_package

CODELLAMA: BundledPackage = make_bundled_package(
    "seiso/codellama_compress",
    "seiso.codellama_compress",
    missing_hint="Expected seiso.codellama_compress",
)
ADAPTIVE_QUANT: BundledPackage = make_bundled_package(
    "seiso/adaptive_quant",
    "seiso.adaptive_quant",
    missing_hint="Expected seiso.adaptive_quant",
)
