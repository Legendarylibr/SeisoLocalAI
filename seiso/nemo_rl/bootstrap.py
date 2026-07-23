"""Locate a NeMo RL checkout and the ``uv`` launcher."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

_ENV_ROOT = "SEISO_NEMO_RL_ROOT"
_MARKER_FILES = (
    "examples/run_grpo.py",
    "nemo_rl",
    "pyproject.toml",
)


def resolve_nemo_rl_root(explicit: str | Path | None = None) -> Path:
    """Resolve the NeMo RL repository root.

    Order: explicit path → ``SEISO_NEMO_RL_ROOT`` → common sibling checkouts.
    """
    candidates: list[Path] = []
    if explicit is not None and str(explicit).strip():
        candidates.append(Path(explicit).expanduser())
    env = os.environ.get(_ENV_ROOT, "").strip()
    if env:
        candidates.append(Path(env).expanduser())
    # Common local layouts next to Seiso (not required).
    here = Path(__file__).resolve()
    repo_root = here.parents[2]  # …/SeisoLocalAI-17
    for sibling in (
        repo_root.parent / "RL",
        repo_root.parent / "nemo-rl",
        repo_root.parent / "NeMo-RL",
        Path.home() / "nemo-rl",
        Path.home() / "RL",
    ):
        candidates.append(sibling)

    seen: set[Path] = set()
    for raw in candidates:
        try:
            root = raw.resolve()
        except OSError:
            continue
        if root in seen:
            continue
        seen.add(root)
        if _looks_like_nemo_rl(root):
            return root

    raise FileNotFoundError(
        "NeMo RL checkout not found. Clone https://github.com/NVIDIA-NeMo/RL "
        "(recursive) and set SEISO_NEMO_RL_ROOT to that directory, or pass "
        "nemo_rl_root in the training config."
    )


def _looks_like_nemo_rl(root: Path) -> bool:
    if not root.is_dir():
        return False
    return all((root / marker).exists() for marker in _MARKER_FILES)


def resolve_uv_executable(explicit: str | None = None) -> str:
    """Return a ``uv`` executable path (required by NeMo RL bare-metal launches)."""
    if explicit and str(explicit).strip():
        path = Path(explicit).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            return str(path.resolve())
        raise FileNotFoundError(f"uv executable not found: {explicit}")
    env = os.environ.get("SEISO_UV", "").strip() or os.environ.get("UV", "").strip()
    if env:
        path = Path(env).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            return str(path.resolve())
    found = shutil.which("uv")
    if found:
        return found
    raise FileNotFoundError(
        "uv is required to launch NeMo RL. Install from https://docs.astral.sh/uv/ "
        "or set SEISO_UV / UV to the executable path."
    )


def nemo_rl_available(*, root: str | Path | None = None) -> bool:
    """True when a NeMo RL checkout can be resolved (does not require ``uv``)."""
    try:
        resolve_nemo_rl_root(root)
        return True
    except FileNotFoundError:
        return False
