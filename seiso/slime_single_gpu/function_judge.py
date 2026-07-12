"""Judge function-level coding tasks via assert / check() harnesses.

Sample schema:
  {
    "reward_name": "assert_tests",
    "prompt": "...",
    "entry_point": "foo",                 # optional
    "code_prefix": "def foo(...):\\n",   # optional starter
    "assert_tests": ["assert foo(1)==2", ...],
    "test_setup": "import math\\n",      # optional
    "check_code": "def check(candidate):\\n ...",  # HumanEval style
    "test_timeout_sec": 3.0,
  }
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

_FORMAT_CREDIT = 0.05
_CODE_FENCE_RE = re.compile(
    r"```(?:python|py)?\s*\n(.*?)```",
    flags=re.IGNORECASE | re.DOTALL,
)


def _extract_code(completion: str) -> str:
    if not completion or not str(completion).strip():
        return ""
    matches = _CODE_FENCE_RE.findall(completion)
    if matches:
        return matches[-1].strip()
    stripped = completion.strip()
    if any(tok in stripped for tok in ("def ", "class ", "return ", "import ")):
        return stripped
    return ""


def assert_tests_reward(completion: str, sample: dict[str, Any]) -> float:
    code = _extract_code(completion)
    if not code:
        # try whole completion
        if any(t in completion for t in ("def ", "class ", "return ")):
            code = completion.strip()
        else:
            return 0.0

    prefix = str(sample.get("code_prefix") or "")
    setup = str(sample.get("test_setup") or sample.get("test_setup_code") or "")
    entry = str(sample.get("entry_point") or "").strip()

    # Prefer explicit assert list
    asserts = sample.get("assert_tests") or sample.get("test_list") or []
    check_code = sample.get("check_code") or sample.get("test") or ""

    try:
        timeout = float(sample.get("test_timeout_sec", 3.0))
    except (TypeError, ValueError):
        timeout = 3.0

    if isinstance(asserts, list) and asserts:
        return _run_asserts(prefix + "\n" + code, setup, [str(a) for a in asserts], timeout)

    if check_code and entry:
        return _run_check(prefix + "\n" + code, str(check_code), entry, timeout)

    # Fallback: try to run check(candidate) if present in check_code without entry
    if check_code:
        # guess entry from def lines
        m = re.findall(r"def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", code)
        if m:
            return _run_check(prefix + "\n" + code, str(check_code), m[0], timeout)

    return _FORMAT_CREDIT


def _run_asserts(code: str, setup: str, asserts: list[str], timeout: float) -> float:
    if not asserts:
        return _FORMAT_CREDIT
    body = [
        setup,
        code,
        "",
        "import sys",
        f"_ASSERTS = {asserts!r}",
        "_passed = 0",
        "for _i, _a in enumerate(_ASSERTS):",
        "    try:",
        "        exec(_a, globals())",
        "        _passed += 1",
        "    except Exception:",
        "        pass",
        "print(f'PASSED:{_passed}/{len(_ASSERTS)}')",
        "sys.exit(0 if _passed == len(_ASSERTS) else 1)",
    ]
    return _exec_pass_rate("\n".join(body), timeout, n_tests=len(asserts))


def _run_check(code: str, check_code: str, entry_point: str, timeout: float) -> float:
    # HumanEval: define candidate then check(candidate)
    body = [
        code,
        "",
        check_code,
        "",
        "import sys",
        f"_fn = globals().get({entry_point!r})",
        "if _fn is None:",
        "    print('PASSED:0/1')",
        "    sys.exit(1)",
        "try:",
        "    check(_fn)",
        "    print('PASSED:1/1')",
        "    sys.exit(0)",
        "except Exception as e:",
        "    print('PASSED:0/1')",
        "    print(type(e).__name__, e)",
        "    sys.exit(1)",
    ]
    return _exec_pass_rate("\n".join(body), timeout, n_tests=1)


def _exec_pass_rate(script: str, timeout: float, *, n_tests: int) -> float:
    if len(script) > 400_000:
        return 0.0
    try:
        with tempfile.TemporaryDirectory(prefix="seiso-fn-") as tmp:
            path = Path(tmp) / "task.py"
            path.write_text(script, encoding="utf-8")
            proc = subprocess.run(
                [sys.executable, str(path)],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
    except (subprocess.TimeoutExpired, OSError, UnicodeError):
        return _FORMAT_CREDIT * 0.5

    text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    m = re.search(r"PASSED:(\d+)/(\d+)", text)
    if m:
        passed, total = int(m.group(1)), int(m.group(2))
        if total <= 0:
            return _FORMAT_CREDIT
        rate = passed / total
        return float(_FORMAT_CREDIT + (1.0 - _FORMAT_CREDIT) * rate)
    # binary from exit code
    if proc.returncode == 0:
        return 1.0
    return _FORMAT_CREDIT
