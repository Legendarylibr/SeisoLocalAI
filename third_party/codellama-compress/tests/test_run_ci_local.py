"""Smoke tests for scripts/run_ci_local.py."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


def test_run_ci_local_help() -> None:
    proc = subprocess.run(
        [sys.executable, str(_REPO_ROOT / "scripts" / "run_ci_local.py"), "--help"],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "--job" in proc.stdout
    assert "--fast" in proc.stdout


def test_run_ci_local_list_documents_jobs() -> None:
    proc = subprocess.run(
        [sys.executable, str(_REPO_ROOT / "scripts" / "run_ci_local.py"), "--list"],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "docs/CI_LOCAL.md" in proc.stdout
    for job in ("test", "imports", "security"):
        assert job in proc.stdout
