"""Bootstrap first-party packages that live under ``seiso``."""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BundledPackage:
    root: Path
    ensure_importable: Callable[[], Path]
    require: Callable[[], None]


def make_bundled_package(
    package_dir: str,
    package_name: str,
    *,
    src_subdir: str | None = None,
    missing_hint: str | None = None,
) -> BundledPackage:
    """Create root / ensure / require helpers for a bundled source package."""
    root = Path(__file__).resolve().parents[2] / package_dir
    hint = missing_hint or f"Expected {package_dir}"

    def ensure_importable() -> Path:
        return ensure_bundled_importable(root, src_subdir=src_subdir)

    def require() -> None:
        require_bundled_package(
            root,
            package_name,
            src_subdir=src_subdir,
            missing_hint=hint,
        )

    return BundledPackage(
        root=root, ensure_importable=ensure_importable, require=require
    )


def ensure_bundled_importable(
    bundle_root: Path, *, src_subdir: str | None = None
) -> Path:
    """Insert the package import root on sys.path if needed."""
    import_root = bundle_root / src_subdir if src_subdir else bundle_root.parent
    root = str(import_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    return import_root


def require_bundled_package(
    bundle_root: Path,
    package_name: str,
    *,
    src_subdir: str | None = None,
    missing_hint: str | None = None,
) -> Path:
    """Ensure a bundled package is importable or raise RuntimeError."""
    import_root = ensure_bundled_importable(bundle_root, src_subdir=src_subdir)
    try:
        __import__(package_name)
    except ImportError as exc:
        hint = (
            missing_hint or f"Expected {import_root / package_name.replace('.', '/')}"
        )
        raise RuntimeError(f"Bundled package {package_name!r} missing. {hint}") from exc
    return import_root
