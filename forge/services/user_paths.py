"""Per-user filesystem path policy — tenant isolation under shared data_dir."""

from __future__ import annotations

from pathlib import Path

from seiso.security import SecurityError, safe_join

_USER_SCOPED_ROOTS = frozenset({"uploads", "knowledge", "artifacts", "sandbox", "models", "checkpoints", "exports"})


def user_dir(sandbox_root: Path, user_id: str, category: str) -> Path:
    """Return (and does not create) a user-owned directory under category."""
    if category not in _USER_SCOPED_ROOTS:
        raise SecurityError(f"Unknown user path category: {category}")
    return safe_join(sandbox_root, category, user_id)


def is_local_filesystem_path(target: str | Path) -> bool:
    """True when *target* refers to a host path rather than a HF hub ID."""
    p = Path(target).expanduser()
    if p.is_absolute():
        return True
    if p.exists():
        return True
    s = str(target)
    return s.startswith(("./", "../", "~/"))


def assert_user_config_file(sandbox_root: Path, user_id: str, config_file: str) -> None:
    """Vendor preset filenames or absolute paths under the user's data tree only."""
    path = Path(config_file).expanduser()
    if path.is_absolute():
        assert_user_path(sandbox_root, user_id, path)
        return
    if "/" in config_file or "\\" in config_file or path.parts != (config_file,):
        raise SecurityError(
            "config_file must be a preset filename or an absolute path under your data directory"
        )


def assert_user_training_config(sandbox_root: Path, user_id: str, config: dict) -> None:
    """Validate local dataset and checkpoint paths are scoped to the requesting user."""
    dataset = config.get("dataset")
    if dataset and is_local_filesystem_path(dataset):
        assert_user_path(sandbox_root, user_id, dataset)
    resume = config.get("resume_from")
    if resume:
        assert_user_path(sandbox_root, user_id, resume)


def assert_user_path(sandbox_root: Path, user_id: str, target: str | Path) -> Path:
    """Path must be inside sandbox and under an allowed root scoped to user_id.

    Inventory symlinks are checked at their logical location under the user tree,
    then resolved for inference file access.
    """
    source = Path(target).expanduser()
    logical = source.absolute()
    base = sandbox_root.resolve()
    try:
        rel = logical.relative_to(base)
    except ValueError as exc:
        raise SecurityError(f"Path must be inside {base}") from exc
    if not rel.parts:
        raise SecurityError("Invalid path")
    root = rel.parts[0]
    if root not in _USER_SCOPED_ROOTS:
        raise SecurityError(f"Access denied to path root: {root!r}")
    if len(rel.parts) < 2 or rel.parts[1] != user_id:
        raise SecurityError(f"Path must be under {root}/{user_id}/")
    if not (source.exists() or source.is_symlink()):
        raise SecurityError(f"Model path not found: {source}")
    resolved = source.resolve()
    if source.is_symlink() and not resolved.exists():
        raise SecurityError(
            f"Model cache link is broken — re-download from Hub: {logical.name}"
        )
    return resolved


def assert_llama_cpp_binary(target: str | Path) -> Path:
    """Ensure llama.cpp binary path is an existing regular file in allowed locations."""
    source = Path(target).expanduser()
    if not source.is_absolute():
        raise SecurityError("llama_cpp_binary must be an absolute path")
    if source.is_symlink():
        raise SecurityError("llama_cpp_binary cannot be a symlink")
    resolved = source.resolve()
    if not resolved.is_file():
        raise SecurityError("llama_cpp_binary must point to an existing file")
    allowed = ("/usr/", "/opt/", "/bin/", "/sbin/", "/Users/", "/home/")
    path_str = str(resolved)
    if not any(path_str.startswith(prefix) for prefix in allowed):
        raise SecurityError("llama_cpp_binary is outside allowed system locations")
    return resolved
