"""Per-user filesystem path policy — tenant isolation under shared data_dir."""

from __future__ import annotations

from pathlib import Path

from seiso.security import SecurityError, assert_within, safe_join

_USER_SCOPED_ROOTS = frozenset(
    {
        "uploads",
        "knowledge",
        "artifacts",
        "sandbox",
        "models",
        "checkpoints",
        "exports",
        "compress",
        "distill_rl",
        "rl_quant",
    }
)
_SHARED_CACHE_ROOTS = frozenset({"hf_cache"})


def _logical_path(source: Path) -> Path:
    """Absolute path of *source* without following its final symlink component."""
    source = source.expanduser()
    parent = source.parent if source.is_absolute() else source.absolute().parent
    parent_resolved = Path.cwd().resolve() if parent == Path(".") else parent.resolve()
    return parent_resolved / source.name


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
    """Allow preset filenames or absolute paths under the user's data tree only."""
    path = Path(config_file).expanduser()
    if path.is_absolute():
        assert_user_path(sandbox_root, user_id, path)
        return
    if "/" in config_file or "\\" in config_file or path.parts != (config_file,):
        raise SecurityError(
            "config_file must be a preset filename or an absolute path under your data directory"
        )


def resolve_training_dataset_path(
    sandbox_root: Path, user_id: str, dataset: str, *, install_root: Path | None = None
) -> str:
    """Map CLI-style relative dataset paths into the user's uploads sandbox."""
    if not dataset or not is_local_filesystem_path(dataset):
        return dataset

    path = Path(dataset).expanduser()
    if path.is_absolute():
        return dataset

    uploads = user_dir(sandbox_root, user_id, "uploads")
    uploads.mkdir(parents=True, exist_ok=True)
    user_copy = uploads / path.name
    if user_copy.exists():
        return str(user_copy)

    if path.name == "sample.jsonl":
        candidates: list[Path] = []
        if install_root is not None:
            candidates.append(install_root / "data" / path.name)
        candidates.append(Path.cwd() / "data" / path.name)
        for bundled in candidates:
            if bundled.is_file():
                user_copy.write_bytes(bundled.read_bytes())
                return str(user_copy)

    return dataset


def assert_user_training_config(sandbox_root: Path, user_id: str, config: dict) -> None:
    """Validate local dataset and checkpoint paths are scoped to the requesting user."""
    dataset = config.get("dataset")
    if dataset and is_local_filesystem_path(dataset):
        assert_user_path(sandbox_root, user_id, dataset)
    resume = config.get("resume_from")
    if resume:
        assert_user_path(sandbox_root, user_id, resume)


def _assert_resolved_scope(
    base: Path,
    user_id: str,
    logical: Path,
    resolved: Path,
) -> None:
    """Ensure resolved target is authorized for this user (blocks symlink escapes)."""
    rel = resolved.relative_to(base)
    if not rel.parts:
        raise SecurityError("Invalid path")

    root = rel.parts[0]
    if root in _USER_SCOPED_ROOTS:
        if len(rel.parts) >= 2 and rel.parts[1] == user_id:
            return
        raise SecurityError(f"Path must be under {root}/{user_id}/")

    if root in _SHARED_CACHE_ROOTS:
        log_rel = logical.relative_to(base)
        if (
            log_rel.parts
            and log_rel.parts[0] == "models"
            and len(log_rel.parts) >= 2
            and log_rel.parts[1] == user_id
        ):
            return
        raise SecurityError(
            "Shared cache paths are only reachable via your model inventory"
        )

    raise SecurityError(f"Access denied to path root: {root!r}")


def assert_user_path(sandbox_root: Path, user_id: str, target: str | Path) -> Path:
    """Path must be inside sandbox and under an allowed root scoped to user_id.

    Inventory symlinks are checked at their logical location under the user tree;
    the resolved target must remain inside the sandbox (no host-path escapes).
    """
    base = sandbox_root.resolve()
    source = Path(target).expanduser()
    logical = _logical_path(source)

    try:
        log_rel = logical.relative_to(base)
    except ValueError as exc:
        raise SecurityError(f"Path must be inside {base}") from exc
    if not log_rel.parts:
        raise SecurityError("Invalid path")
    log_root = log_rel.parts[0]
    if log_root not in _USER_SCOPED_ROOTS:
        raise SecurityError(f"Access denied to path root: {log_root!r}")
    if len(log_rel.parts) < 2 or log_rel.parts[1] != user_id:
        raise SecurityError(f"Path must be under {log_root}/{user_id}/")
    if not (source.exists() or source.is_symlink()):
        raise SecurityError(f"Model path not found: {source}")

    resolved = assert_within(base, source.resolve())
    if source.is_symlink() and not resolved.exists():
        raise SecurityError(
            f"Model cache link is broken — re-download from Hub: {logical.name}"
        )
    _assert_resolved_scope(base, user_id, logical, resolved)
    return resolved


def assert_user_download_file(
    sandbox_root: Path,
    user_id: str,
    file_path: str | Path,
    *,
    container_dir: Path | None = None,
) -> Path:
    """Return a sandbox-scoped file safe to stream via FileResponse.

    Re-validates the resolved path so directory scans and metadata joins cannot
    escape the per-user sandbox via symlinks or absolute path segments.
    """
    source = Path(file_path)
    if container_dir is not None:
        container = assert_user_path(sandbox_root, user_id, container_dir)
        if source.is_absolute():
            raise SecurityError("Download file must be relative to the model directory")
        candidate = (container / source).resolve()
        try:
            candidate.relative_to(container.resolve())
        except ValueError as exc:
            raise SecurityError(
                "Download file must be inside the model directory"
            ) from exc
        source = candidate
    return assert_user_path(sandbox_root, user_id, source)


def pick_user_download_file(
    sandbox_root: Path,
    user_id: str,
    directory: Path,
    *,
    pattern: str = "*.gguf",
    relative_name: str | None = None,
) -> Path:
    """Pick the first sandbox-safe file under *directory* for download."""
    from seiso.io.files import iter_matching_files

    container = assert_user_path(sandbox_root, user_id, directory)
    if container.is_file():
        return container

    if relative_name is not None:
        if Path(relative_name).is_absolute():
            raise SecurityError("Download file must be a relative path")
        return assert_user_download_file(
            sandbox_root,
            user_id,
            relative_name,
            container_dir=container,
        )

    for match in iter_matching_files(container, pattern):
        return assert_user_path(sandbox_root, user_id, match)
    raise SecurityError(f"No downloadable file matching {pattern!r}")


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
    path_str = str(resolved)
    system_prefixes = ("/usr/", "/opt/", "/bin/", "/sbin/", "/Users/", "/home/")
    parts_lower = {part.lower() for part in resolved.parts}
    if not (
        any(path_str.startswith(prefix) for prefix in system_prefixes)
        or ".venv" in parts_lower
        or "venv" in parts_lower
    ):
        raise SecurityError("llama_cpp_binary is outside allowed system locations")
    return resolved
