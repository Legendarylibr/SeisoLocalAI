"""Bootstrap vendored packages onto sys.path."""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VendorBootstrap:
    root: Path
    ensure_importable: Callable[[], Path]
    require: Callable[[], None]


def make_vendor_bootstrap(
    vendor_dir: str,
    package_name: str,
    *,
    src_subdir: str | None = "src",
    missing_hint: str | None = None,
) -> VendorBootstrap:
    """Create vendor_root / ensure / require helpers for a third_party tree."""
    root = Path(__file__).resolve().parents[2] / "third_party" / vendor_dir
    hint = missing_hint or f"Expected third_party/{vendor_dir}"

    def ensure_importable() -> Path:
        return ensure_vendor_importable(root, src_subdir=src_subdir)

    def require() -> None:
        require_vendor_package(
            root,
            package_name,
            src_subdir=src_subdir,
            missing_hint=hint,
        )

    return VendorBootstrap(root=root, ensure_importable=ensure_importable, require=require)


def ensure_vendor_importable(vendor_root: Path, *, src_subdir: str | None = "src") -> Path:
    """Insert the vendor import root on sys.path if needed."""
    import_root = vendor_root / src_subdir if src_subdir else vendor_root
    root = str(import_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    return import_root


def require_vendor_package(
    vendor_root: Path,
    package_name: str,
    *,
    src_subdir: str | None = "src",
    missing_hint: str | None = None,
) -> Path:
    """Ensure a vendored package is importable or raise RuntimeError."""
    import_root = ensure_vendor_importable(vendor_root, src_subdir=src_subdir)
    try:
        __import__(package_name)
    except ImportError as exc:
        hint = missing_hint or f"Expected {import_root / package_name.replace('.', '/')}"
        raise RuntimeError(f"Vendored package {package_name!r} missing. {hint}") from exc
    return import_root
