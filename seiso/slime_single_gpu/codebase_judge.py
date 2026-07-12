"""Judge multi-file synthetic codebases via pytest (or custom test scripts).

Sample schema (JSONL row):
  {
    "prompt": "...",
    "reward_name": "codebase_tests",
    "domain": "codebase",
    "codebase": {
      "files": {"src/pkg/__init__.py": "", "src/pkg/core.py": "...", ...},
      "tests": {"tests/test_core.py": "..."},
      "target_files": ["src/pkg/core.py"],   # files the model should write/replace
      "hidden_files": {...},                 # optional, never shown in prompt
      "setup": [],                           # optional shell setup (unused by default)
    },
    "test_timeout_sec": 8,
    "pytest_args": ["-q", "--tb=no"],
  }

Model completions may provide:
  - path-tagged fences: ```python path=src/pkg/core.py
  - bare ```python (applied to the first target file)
  - multiple fences for multi-file edits
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

_PATH_FENCE_RE = re.compile(
    r"```(?:python|py)?\s*(?:path\s*=\s*|file\s*=\s*|#\s*file:\s*)([^\n`]+)\n(.*?)```",
    flags=re.IGNORECASE | re.DOTALL,
)
_PLAIN_FENCE_RE = re.compile(
    r"```(?:python|py)?\s*\n(.*?)```",
    flags=re.IGNORECASE | re.DOTALL,
)
_DEFAULT_TIMEOUT = 8.0
_FORMAT_CREDIT = 0.05


def extract_file_map(completion: str, target_files: list[str] | None = None) -> dict[str, str]:
    """Parse multi-file or single-file python fences from a completion."""
    if not completion or not str(completion).strip():
        return {}
    files: dict[str, str] = {}
    for path, body in _PATH_FENCE_RE.findall(completion):
        p = str(path).strip().strip("\"'` ")
        if p:
            files[p] = body.strip() + ("\n" if body.strip() and not body.endswith("\n") else "")
    if files:
        return files
    plains = _PLAIN_FENCE_RE.findall(completion)
    if not plains:
        # Heuristic: whole completion looks like a module.
        stripped = completion.strip()
        if any(tok in stripped for tok in ("def ", "class ", "import ", "from ")):
            if target_files:
                return {target_files[0]: stripped + "\n"}
        return {}
    if target_files:
        # Map last N plain fences onto last N targets (or first if one).
        if len(plains) == 1 or len(target_files) == 1:
            return {target_files[0]: plains[-1].strip() + "\n"}
        out: dict[str, str] = {}
        for path, body in zip(target_files, plains[-len(target_files) :], strict=False):
            out[path] = body.strip() + "\n"
        return out
    return {"solution.py": plains[-1].strip() + "\n"}


def codebase_tests_reward(completion: str, sample: dict[str, Any]) -> float:
    """Score a completion by installing edits into a scaffold and running pytest."""
    codebase = sample.get("codebase") or {}
    if not isinstance(codebase, dict):
        return 0.0
    files = dict(codebase.get("files") or {})
    tests = dict(codebase.get("tests") or {})
    hidden = dict(codebase.get("hidden_files") or {})
    targets = list(codebase.get("target_files") or [])
    if not files and not tests:
        return 0.0

    edits = extract_file_map(completion, targets or list(files.keys()))
    if not edits:
        return 0.0

    # Format credit for producing parseable code.
    format_credit = float(sample.get("format_credit", _FORMAT_CREDIT))

    # Apply model edits (only allow writing into known tree or targets).
    allowed = set(files) | set(targets) | set(hidden) | set(tests)
    applied = 0
    for path, body in edits.items():
        # Normalize path
        path = path.lstrip("./")
        if allowed and path not in allowed and not any(
            path.endswith(t.split("/")[-1]) for t in targets
        ):
            # Allow basename match onto a unique target.
            base = Path(path).name
            matches = [t for t in targets if Path(t).name == base]
            if len(matches) == 1:
                path = matches[0]
            else:
                continue
        files[path] = body
        applied += 1
    if applied == 0:
        return format_credit * 0.5

    try:
        timeout = float(sample.get("test_timeout_sec", _DEFAULT_TIMEOUT))
    except (TypeError, ValueError):
        timeout = _DEFAULT_TIMEOUT
    pytest_args = sample.get("pytest_args") or ["-q", "--tb=line"]
    if not isinstance(pytest_args, list):
        pytest_args = ["-q", "--tb=line"]

    result = _run_project(
        scaffold_files={**files, **hidden},
        test_files=tests,
        timeout=timeout,
        pytest_args=[str(a) for a in pytest_args],
    )
    if result["n_tests"] <= 0:
        # Ran but collected nothing — weak credit if project imported.
        return format_credit if result["exit_code"] == 0 else format_credit * 0.25

    pass_rate = result["passed"] / max(1, result["n_tests"])
    # Soft weight: applying the right files matters a little.
    target_hit = 1.0
    if targets:
        hit = sum(1 for t in targets if t in edits or Path(t).name in {Path(p).name for p in edits})
        target_hit = hit / len(targets)
    shaped = format_credit + (1.0 - format_credit) * (0.85 * pass_rate + 0.15 * target_hit * pass_rate)
    return float(max(0.0, min(1.0, shaped)))


def _run_project(
    *,
    scaffold_files: dict[str, str],
    test_files: dict[str, str],
    timeout: float,
    pytest_args: list[str],
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="seiso-cb-") as tmp:
        root = Path(tmp)
        for rel, content in {**scaffold_files, **test_files}.items():
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content if content.endswith("\n") else content + "\n", encoding="utf-8")

        # Ensure packages importable.
        for rel in list(scaffold_files) + list(test_files):
            parts = Path(rel).parts
            if "src" in parts:
                # mark src as path root via conftest/sitecustomize
                pass
            # create __init__.py for intermediate dirs under src/
            p = root / rel
            for parent in p.parents:
                if parent == root:
                    break
                if parent.name in {"tests", "src"} or parent.name.startswith("."):
                    continue
                init = parent / "__init__.py"
                if parent.exists() and not init.exists():
                    # only if sibling .py modules exist
                    if any(parent.glob("*.py")):
                        init.write_text("", encoding="utf-8")

        conftest = root / "conftest.py"
        if not conftest.exists():
            conftest.write_text(
                "import sys\n"
                "from pathlib import Path\n"
                "ROOT = Path(__file__).resolve().parent\n"
                "src = ROOT / 'src'\n"
                "if src.exists():\n"
                "    sys.path.insert(0, str(src))\n"
                "sys.path.insert(0, str(ROOT))\n",
                encoding="utf-8",
            )

        cmd = [sys.executable, "-m", "pytest", *pytest_args]
        try:
            proc = subprocess.run(
                cmd,
                cwd=root,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            return {
                "exit_code": 124,
                "passed": 0,
                "failed": 0,
                "n_tests": 0,
                "stdout": "",
                "stderr": str(exc),
            }

        passed, failed, n_tests = _parse_pytest_counts(proc.stdout + "\n" + proc.stderr)
        return {
            "exit_code": int(proc.returncode),
            "passed": passed,
            "failed": failed,
            "n_tests": n_tests if n_tests > 0 else (passed + failed),
            "stdout": proc.stdout[-4000:],
            "stderr": proc.stderr[-2000:],
        }


def _parse_pytest_counts(text: str) -> tuple[int, int, int]:
    """Parse pytest -q summary lines."""
    # Examples: "5 passed in 0.01s", "2 failed, 3 passed", "1 passed, 1 warning"
    passed = failed = 0
    m = re.search(r"(\d+)\s+passed", text)
    if m:
        passed = int(m.group(1))
    m = re.search(r"(\d+)\s+failed", text)
    if m:
        failed = int(m.group(1))
    m = re.search(r"(\d+)\s+error", text)
    errors = int(m.group(1)) if m else 0
    failed += errors
    n = passed + failed
    if n == 0:
        # collection error etc.
        if "error" in text.lower() or "failed" in text.lower():
            return 0, 1, 1
    return passed, failed, n


def gold_passes(sample: dict[str, Any]) -> bool:
    """Verify the sample's gold target files pass tests (dataset QA)."""
    codebase = sample.get("codebase") or {}
    files = dict(codebase.get("files") or {})
    gold = dict(codebase.get("gold_files") or {})
    files.update(gold)
    tests = dict(codebase.get("tests") or {})
    hidden = dict(codebase.get("hidden_files") or {})
    result = _run_project(
        scaffold_files={**files, **hidden},
        test_files=tests,
        timeout=float(sample.get("test_timeout_sec", _DEFAULT_TIMEOUT)),
        pytest_args=["-q", "--tb=no"],
    )
    return result["n_tests"] > 0 and result["failed"] == 0 and result["passed"] == result["n_tests"]
