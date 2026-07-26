"""Verify dependency lockfiles against a separate SHA-256 digest manifest."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from pathlib import Path
from typing import Any, cast

from seiso.security import SecurityError

DEFAULT_DIGESTS_REL = Path("locks/digests.json")
DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PYPROJECT_REL = Path("pyproject.toml")
DEFAULT_PYTHON_LOCK_REL = Path("locks/python.lock")

_HASH_LINE = re.compile(r"^\s+--hash=sha256:")
_PACKAGE_LINE = re.compile(r"^[a-zA-Z0-9][\w.\-]*==")
_LOCKED_VERSION = re.compile(
    r"^(?P<name>[a-zA-Z0-9][\w.\-]*)==(?P<version>[^\s\\]+)",
    re.MULTILINE,
)

# Declared floors for CVEs that must stay aligned across pyproject + lock.
# (package, minimum version, where the floor must be declared)
SECURITY_FLOOR_CHECKS: tuple[tuple[str, str, str], ...] = (
    # CVE-2026-59885 / CVE-2026-59886 — pyasn1 REAL/OID DoS
    ("pyasn1", "0.6.4", "forge"),
    # CVE-2026-59890 — setuptools MANIFEST.in exclusion bypass
    ("setuptools", "83.0.0", "build-system"),
)

# Extras compiled into locks/python.lock (see scripts/update_dep_locks.py).
LOCKED_EXTRAS: tuple[str, ...] = ("forge", "train", "dev")

# Marker environment for lock coverage (matches GitHub Actions ubuntu runners).
_LOCK_MARKER_ENV: dict[str, str] = {
    "os_name": "posix",
    "sys_platform": "linux",
    "platform_system": "Linux",
    "platform_machine": "x86_64",
    "python_version": "3.10",
    "python_full_version": "3.10.0",
    "implementation_name": "cpython",
}


class LockDigestError(SecurityError):
    """Raised when a lockfile fails digest or hash-policy verification."""


def sha256_file(path: Path, *, max_bytes: int | None = None) -> str:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            size += len(chunk)
            if max_bytes is not None and size > max_bytes:
                raise ValueError(f"File exceeds hash limit: {path}")
            digest.update(chunk)
    return digest.hexdigest()


def load_digest_manifest(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise LockDigestError("Digest manifest must be a JSON object")
    data: dict[str, Any] = raw
    if data.get("algorithm") != "sha256":
        raise LockDigestError(
            f"Unsupported digest algorithm: {data.get('algorithm')!r}"
        )
    artifacts = data.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise LockDigestError("Digest manifest missing non-empty 'artifacts' object")
    return data


def verify_lock_digests(
    *,
    repo_root: Path | None = None,
    digests_path: Path | None = None,
) -> list[str]:
    """Verify tracked lockfiles match digests in the separate manifest."""
    root = (repo_root or DEFAULT_REPO_ROOT).resolve()
    manifest_path = (digests_path or (root / DEFAULT_DIGESTS_REL)).resolve()
    if not manifest_path.is_file():
        raise LockDigestError(f"Digest manifest not found: {manifest_path}")

    expected_by_path = load_digest_manifest(manifest_path)["artifacts"]
    verified: list[str] = []

    for rel_path, expected_digest in expected_by_path.items():
        if not isinstance(rel_path, str) or not isinstance(expected_digest, str):
            raise LockDigestError("Digest manifest entries must be strings")
        if len(expected_digest) != 64 or any(
            ch not in "0123456789abcdef" for ch in expected_digest
        ):
            raise LockDigestError(f"Invalid SHA-256 digest for {rel_path!r}")

        lock_path = (root / rel_path).resolve()
        if not lock_path.is_file():
            raise LockDigestError(f"Lockfile missing: {lock_path}")

        actual_digest = sha256_file(lock_path)
        if not hmac.compare_digest(actual_digest, expected_digest):
            raise LockDigestError(
                f"Digest mismatch for {rel_path}: "
                f"expected {expected_digest}, computed {actual_digest}"
            )
        verified.append(rel_path)

    return verified


def verify_python_lock_has_hashes(lock_path: Path) -> None:
    """Require every pinned package in a pip-compile lock to declare sha256 hashes."""
    if not lock_path.is_file():
        raise LockDigestError(f"Python lockfile missing: {lock_path}")

    current_package: str | None = None
    saw_hash = False
    missing_hashes: list[str] = []

    with lock_path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip()
            if not line or line.startswith("#"):
                continue

            if _PACKAGE_LINE.match(line):
                if current_package is not None and not saw_hash:
                    missing_hashes.append(current_package)
                current_package = line.split()[0]
                saw_hash = _HASH_LINE.match(line) is not None
                continue

            if _HASH_LINE.match(line):
                saw_hash = True
                continue

            if current_package is not None and not line.startswith(" "):
                if not saw_hash:
                    missing_hashes.append(current_package)
                current_package = None
                saw_hash = False

    if current_package is not None and not saw_hash:
        missing_hashes.append(current_package)

    if missing_hashes:
        sample = ", ".join(missing_hashes[:5])
        suffix = "..." if len(missing_hashes) > 5 else ""
        raise LockDigestError(
            f"{len(missing_hashes)} package(s) in {lock_path.name} lack sha256 hashes: "
            f"{sample}{suffix}"
        )


def _load_toml(path: Path) -> dict[str, Any]:
    try:
        import tomllib
    except ModuleNotFoundError:  # Python < 3.11
        try:
            import tomli as tomllib
        except ModuleNotFoundError as exc:
            raise LockDigestError(
                "tomllib/tomli required to verify security floors in pyproject.toml"
            ) from exc
    raw: object = tomllib.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise LockDigestError(f"TOML root must be a table: {path}")
    return cast(dict[str, Any], raw)


def _version_tuple(version: str) -> tuple[int, ...]:
    parts: list[int] = []
    for piece in version.split("."):
        digits = ""
        for char in piece:
            if char.isdigit():
                digits += char
            else:
                break
        parts.append(int(digits or 0))
    return tuple(parts)


def _version_gte(left: tuple[int, ...], right: tuple[int, ...]) -> bool:
    width = max(len(left), len(right))
    left_p = left + (0,) * (width - len(left))
    right_p = right + (0,) * (width - len(right))
    return left_p >= right_p


def locked_package_versions(lock_path: Path) -> dict[str, str]:
    """Return ``{normalized_name: version}`` for ``name==version`` pins in a lockfile."""
    if not lock_path.is_file():
        raise LockDigestError(f"Python lockfile missing: {lock_path}")
    text = lock_path.read_text(encoding="utf-8")
    versions: dict[str, str] = {}
    for match in _LOCKED_VERSION.finditer(text):
        versions[match.group("name").lower()] = match.group("version")
    return versions


def _requirement_name(req: str) -> str:
    return re.split(r"[<>=!~;\[]", req, maxsplit=1)[0].strip().lower()


def _requirement_min_version(req: str) -> str | None:
    """Best-effort minimum from a ``pkg>=X`` / ``pkg==X`` requirement string."""
    match = re.search(r"(?:>=|==)\s*([0-9]+(?:\.[0-9]+)*)", req)
    return match.group(1) if match else None


def verify_security_floors(
    *,
    repo_root: Path | None = None,
    pyproject_path: Path | None = None,
    lock_path: Path | None = None,
) -> list[str]:
    """Ensure CVE floors are declared in pyproject and satisfied by the Python lock."""
    root = (repo_root or DEFAULT_REPO_ROOT).resolve()
    project_path = (pyproject_path or (root / DEFAULT_PYPROJECT_REL)).resolve()
    python_lock = (lock_path or (root / DEFAULT_PYTHON_LOCK_REL)).resolve()
    if not project_path.is_file():
        raise LockDigestError(f"pyproject.toml missing: {project_path}")

    cfg = _load_toml(project_path)
    build_requires = list(cfg.get("build-system", {}).get("requires", []))
    extras = dict(cfg.get("project", {}).get("optional-dependencies", {}))
    locked = locked_package_versions(python_lock)
    verified: list[str] = []

    for package, minimum, location in SECURITY_FLOOR_CHECKS:
        min_tuple = _version_tuple(minimum)
        if location == "build-system":
            declared = [r for r in build_requires if _requirement_name(r) == package]
            if not declared:
                raise LockDigestError(
                    f"{package}>={minimum} missing from [build-system].requires"
                )
        else:
            declared = [
                r for r in extras.get(location, []) if _requirement_name(r) == package
            ]
            if not declared:
                raise LockDigestError(
                    f"{package}>={minimum} missing from optional-dependencies.{location}"
                )

        declared_mins = [_requirement_min_version(r) for r in declared]
        if not any(
            dm is not None and _version_gte(_version_tuple(dm), min_tuple)
            for dm in declared_mins
        ):
            raise LockDigestError(
                f"{package} floor in {location} must be >={minimum}; found {declared!r}"
            )

        locked_version = locked.get(package)
        if locked_version is None:
            raise LockDigestError(f"{package} missing from {python_lock.name}")
        if not _version_gte(_version_tuple(locked_version), min_tuple):
            raise LockDigestError(
                f"{package}=={locked_version} in {python_lock.name} is below floor {minimum}"
            )
        verified.append(f"{package}>={minimum} ({location}, lock={locked_version})")

    return verified


def verify_lock_covers_pyproject(
    *,
    repo_root: Path | None = None,
    pyproject_path: Path | None = None,
    lock_path: Path | None = None,
) -> list[str]:
    """Ensure every locked extra / runtime requirement is pinned to a satisfying version."""
    try:
        from packaging.requirements import Requirement
        from packaging.utils import canonicalize_name
        from packaging.version import Version
    except ModuleNotFoundError as exc:
        raise LockDigestError(
            "packaging required to verify pyproject coverage against the lock"
        ) from exc

    root = (repo_root or DEFAULT_REPO_ROOT).resolve()
    project_path = (pyproject_path or (root / DEFAULT_PYPROJECT_REL)).resolve()
    python_lock = (lock_path or (root / DEFAULT_PYTHON_LOCK_REL)).resolve()
    if not project_path.is_file():
        raise LockDigestError(f"pyproject.toml missing: {project_path}")

    cfg = _load_toml(project_path)
    extras = dict(cfg.get("project", {}).get("optional-dependencies", {}))
    locked_raw = locked_package_versions(python_lock)
    locked = {canonicalize_name(name): version for name, version in locked_raw.items()}
    checked: list[str] = []

    req_strings: list[tuple[str, str]] = [
        ("dependencies", value)
        for value in cfg.get("project", {}).get("dependencies", [])
    ]
    for extra in LOCKED_EXTRAS:
        for value in extras.get(extra, []):
            req_strings.append((f"optional-dependencies.{extra}", value))

    for location, raw in req_strings:
        req = Requirement(raw)
        name = canonicalize_name(req.name)
        locked_version = locked.get(name)
        if locked_version is None:
            # Platform-gated extras may be absent until a universal lock refresh.
            if req.marker is not None:
                continue
            raise LockDigestError(
                f"{req.name} from {location} missing from {python_lock.name}"
            )
        if req.marker is not None and not req.marker.evaluate(_LOCK_MARKER_ENV):
            # Present in a universal lock but inactive on this platform — still OK.
            checked.append(f"{req.name}=={locked_version} ({location}, inactive-marker)")
            continue
        if req.specifier and Version(locked_version) not in req.specifier:
            raise LockDigestError(
                f"{req.name}=={locked_version} in {python_lock.name} "
                f"does not satisfy {req.specifier} ({location})"
            )
        checked.append(f"{req.name}=={locked_version} ({location})")

    return checked


def build_digest_manifest(
    repo_root: Path, artifacts: dict[str, Path]
) -> dict[str, Any]:
    manifest_artifacts: dict[str, str] = {}
    for rel_path, absolute_path in artifacts.items():
        manifest_artifacts[rel_path] = sha256_file(absolute_path)
    return {
        "version": 1,
        "algorithm": "sha256",
        "artifacts": manifest_artifacts,
    }


def write_digest_manifest(manifest: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, indent=2, sort_keys=False) + "\n"
    path.write_text(payload, encoding="utf-8")
