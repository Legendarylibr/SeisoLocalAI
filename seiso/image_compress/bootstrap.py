"""Bootstrap vendored sd_compress onto sys.path."""

from __future__ import annotations

from pathlib import Path

from seiso.vendor.bootstrap import make_vendor_bootstrap

_bs = make_vendor_bootstrap(
    "sd-distill-prune-quant",
    "sd_compress",
    src_subdir=None,
    missing_hint="Expected third_party/sd-distill-prune-quant/sd_compress",
)
_VENDOR_ROOT = _bs.root


def vendor_root() -> Path:
    return _VENDOR_ROOT


ensure_sd_compress_importable = _bs.ensure_importable
require_sd_compress = _bs.require
