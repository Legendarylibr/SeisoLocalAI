"""Per-user filesystem path policy — tenant isolation under shared data_dir.

Scoped root names come from ``seiso.security.USER_SCOPED_DATA_ROOTS``. This
module is the Forge wrapper that adds inventory→hf_cache symlink allowance
(``assert_user_path``) on top of the shared root set.
"""

from __future__ import annotations

from pathlib import Path

from seiso.security import (
    USER_SCOPED_DATA_ROOTS,
    SecurityError,
    assert_within,
    safe_join,
)

# Shared HF cache is not user-scoped; reachable only via models/<user_id>/ inventory links.
_SHARED_CACHE_ROOTS = frozenset({"hf_cache"})


def _logical_path(source: Path) -> Path:
    """Absolute path of *source* without following its final symlink component."""
    source = source.expanduser()
    parent = source.parent if source.is_absolute() else source.absolute().parent
    parent_resolved = Path.cwd().resolve() if parent == Path(".") else parent.resolve()
    return parent_resolved / source.name


def user_dir(sandbox_root: Path, user_id: str, category: str) -> Path:
    """Return (and does not create) a user-owned directory under category."""
    if category not in USER_SCOPED_DATA_ROOTS:
        raise SecurityError(f"Unknown user path category: {category}")
    return safe_join(sandbox_root, category, user_id)


def is_local_filesystem_path(target: str | Path) -> bool:
    """True when *target* refers to a host path rather than a HF hub ID.

    Kept in sync with ``seiso.training.datasets.looks_like_local_dataset_path``
    so Forge gates and materialize sandbox checks agree on relative ``*.jsonl``.
    """
    from seiso.training.datasets import looks_like_local_dataset_path

    return looks_like_local_dataset_path(target)


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
    """Map CLI-style relative dataset paths into the user sandbox.

    Search order matches ``assert_user_path`` / ``USER_SCOPED_DATA_ROOTS`` so
    relative paths can resolve under uploads, checkpoints, knowledge, etc.
    (S1-004). Bare filenames still prefer uploads (and may copy bundled samples).
    """
    if not dataset or not is_local_filesystem_path(dataset):
        return dataset

    path = Path(dataset).expanduser()
    if path.is_absolute():
        return dataset

    from seiso.security import USER_SCOPED_DATA_ROOTS, SecurityError, assert_within

    rel_parts = [p for p in path.parts if p not in (".",)]
    # Prefer uploads for bare filenames (historical CLI UX).
    search_roots = ("uploads",) + tuple(
        sorted(r for r in USER_SCOPED_DATA_ROOTS if r != "uploads")
    )
    for category in search_roots:
        try:
            root = user_dir(sandbox_root, user_id, category)
        except Exception:
            continue
        if not rel_parts:
            continue
        candidate = root.joinpath(*rel_parts)
        try:
            if candidate.is_file():
                assert_within(root, candidate)
                return str(candidate.resolve())
        except (OSError, ValueError, SecurityError):
            continue

    uploads = user_dir(sandbox_root, user_id, "uploads")
    uploads.mkdir(parents=True, exist_ok=True)
    user_copy = uploads / path.name

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
    eval_ds = config.get("slime_eval_dataset") or config.get("eval_dataset")
    if eval_ds and is_local_filesystem_path(eval_ds):
        assert_user_path(sandbox_root, user_id, eval_ds)
    dataset_ref = config.get("dataset_ref") or config.get("hf_dataset")
    if dataset_ref and is_local_filesystem_path(dataset_ref):
        assert_user_path(sandbox_root, user_id, dataset_ref)
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
    if root in USER_SCOPED_DATA_ROOTS:
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
    if log_root not in USER_SCOPED_DATA_ROOTS:
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


def assert_rl_quant_input_paths(
    sandbox_root: Path,
    user_id: str,
    paths: dict[str, str],
) -> None:
    """Validate merged RL-quant filesystem inputs after config_file/preset load.

    ``prompt_library_path`` may live under the adaptive_quant bundle (presets) or
    the caller's user tree. Binaries use :func:`assert_llama_cpp_binary`.
    """
    from seiso.rl_quant.bootstrap import bundle_root
    from seiso.rl_quant.config_builder import (
        RL_QUANT_BINARY_PATH_KEYS,
        RL_QUANT_DATA_PATH_KEYS,
    )
    from seiso.security import assert_user_scoped_path, assert_within

    bundle = bundle_root().resolve()
    for key, raw in paths.items():
        if key in RL_QUANT_BINARY_PATH_KEYS:
            assert_llama_cpp_binary(raw)
            continue
        if key not in RL_QUANT_DATA_PATH_KEYS:
            continue
        path = Path(raw).expanduser()
        if key == "prompt_library_path":
            candidate = path if path.is_absolute() else bundle / path
            try:
                assert_within(bundle, candidate.resolve())
                continue
            except (OSError, SecurityError):
                pass
        if path.exists() or path.is_symlink():
            assert_user_path(sandbox_root, user_id, path)
        elif path.is_absolute():
            assert_user_scoped_path(sandbox_root, user_id, path)
        else:
            raise SecurityError(
                f"{key} must be an absolute path under your data directory "
                f"or a bundle-relative prompt library"
            )


def assert_llama_cpp_binary(target: str | Path) -> Path:
    """Ensure llama.cpp binary path is an existing regular file in allowed locations."""
    import os

    source = Path(target).expanduser()
    if not source.is_absolute():
        raise SecurityError("llama_cpp_binary must be an absolute path")
    if source.is_symlink():
        raise SecurityError("llama_cpp_binary cannot be a symlink")
    resolved = source.resolve()
    if not resolved.is_file():
        raise SecurityError("llama_cpp_binary must point to an existing file")
    path_str = str(resolved)
    parts_lower = {part.lower() for part in resolved.parts}
    in_venv = ".venv" in parts_lower or "venv" in parts_lower
    banned_prefixes = ("/tmp/", "/var/tmp/", "/private/tmp/", "/dev/", "/proc/")
    if any(path_str.startswith(prefix) for prefix in banned_prefixes) and not in_venv:
        raise SecurityError("llama_cpp_binary cannot be under temporary or device paths")
    # Do not allow arbitrary home-tree binaries (~/.seiso payloads, etc.).
    system_prefixes = ("/usr/", "/opt/", "/bin/", "/sbin/")
    extra_raw = os.environ.get("ADAPTIVE_RL_LLAMA_CPP_BINARY_PREFIXES", "").strip()
    extra_prefixes = tuple(
        p if p.endswith("/") else f"{p}/"
        for p in (part.strip() for part in extra_raw.split(os.pathsep))
        if p
    )
    allowed = system_prefixes + extra_prefixes
    if not (any(path_str.startswith(prefix) for prefix in allowed) or in_venv):
        raise SecurityError(
            "llama_cpp_binary is outside allowed system locations "
            "(/usr|/opt|/bin|/sbin, a venv, or ADAPTIVE_RL_LLAMA_CPP_BINARY_PREFIXES)"
        )
    return resolved
