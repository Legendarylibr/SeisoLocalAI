"""Best-effort restricted Python execution — AST deny-list + resource limits.

This is **not** a full OS sandbox (no network/FS namespace isolation). Prefer
keeping ``allow_code_exec`` off, especially with ``allow_remote``. Limits are
best-effort and may be unavailable on some hosts.
"""

from __future__ import annotations

import ast
import contextlib
import json
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

from forge.security.audit import audit_event
from forge.tools.code_exec_policy import (
    _BLOCKED_ATTRS,
    _BLOCKED_NAMES,
    _MAX_ALLOC_SIZE,
    _MAX_CODE_LEN,
    _MAX_OUTPUT,
    _TIMEOUT_SEC,
    import_blocked,
)
from forge.tools.code_exec_sandbox import (
    assign_windows_job,
    subprocess_limits,
    windows_job_limits,
)
from seiso.security import resolve_data_dir


def _alloc_size(node: ast.AST) -> int | None:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            return None
        if isinstance(node.value, int):
            return abs(node.value)
        if isinstance(node.value, (str, bytes, list, tuple)):
            return len(node.value)
    # ``[0] * N`` / ``(0,) * N`` — sized sequence literal times an integer.
    if isinstance(node, (ast.List, ast.Tuple)):
        return max(len(node.elts), 1)
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, ast.Mult):
            left = _alloc_size(node.left)
            right = _alloc_size(node.right)
            if left is not None and right is not None:
                try:
                    return int(left * right)
                except OverflowError:
                    return _MAX_ALLOC_SIZE + 1
        if isinstance(node.op, ast.Pow):
            base = _alloc_size(node.left)
            exp = _alloc_size(node.right)
            if base is not None and exp is not None and exp >= 0:
                try:
                    return int(base**exp)
                except OverflowError:
                    return _MAX_ALLOC_SIZE + 1
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _alloc_size(node.operand)
        if inner is not None:
            return inner
    return None


def _check_alloc_size(validator: _CodeValidator, node: ast.AST) -> None:
    size = _alloc_size(node)
    if size is not None and size > _MAX_ALLOC_SIZE:
        validator.errors.append(f"Allocation too large (>{_MAX_ALLOC_SIZE} elements)")


class _CodeValidator(ast.NodeVisitor):
    def __init__(self) -> None:
        self.errors: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if import_blocked(alias.name):
                self.errors.append(f"Import blocked: {alias.name}")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module and import_blocked(node.module):
            self.errors.append(f"Import blocked: {node.module}")

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load) and node.id in _BLOCKED_NAMES:
            self.errors.append(f"Name blocked: {node.id}")
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, (ast.Mult, ast.Pow)):
            _check_alloc_size(self, node)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name):
            if node.func.id in _BLOCKED_NAMES:
                self.errors.append(f"Call blocked: {node.func.id}()")
            elif node.func.id == "range" and node.args:
                _check_alloc_size(self, node.args[0])
            elif node.func.id == "list" and node.args:
                arg = node.args[0]
                if (
                    isinstance(arg, ast.Call)
                    and isinstance(arg.func, ast.Name)
                    and arg.func.id == "range"
                    and arg.args
                ):
                    _check_alloc_size(self, arg.args[0])
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("__") and node.attr.endswith("__"):
            self.errors.append(f"Dunder access blocked: {node.attr}")
        elif node.attr in _BLOCKED_ATTRS:
            self.errors.append(f"Attribute access blocked: {node.attr}")
        self.generic_visit(node)


def _validate_code(code: str) -> str | None:
    import unicodedata

    code = unicodedata.normalize("NFKC", code)
    if len(code) > _MAX_CODE_LEN:
        return f"Code exceeds {_MAX_CODE_LEN} characters"
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"Syntax error: {exc}"
    validator = _CodeValidator()
    validator.visit(tree)
    if validator.errors:
        return "; ".join(validator.errors[:5])
    return None


def execute_code(
    code: str, sandbox_root: str | None = None, user_id: str | None = None
) -> str:
    """Run user code in isolated subprocess with AST pre-check."""
    err = _validate_code(code)
    if err:
        return json.dumps({"error": err})

    audit_event("code_exec", code_len=len(code), user_id=user_id)

    root = Path(sandbox_root) if sandbox_root else resolve_data_dir()
    if user_id:
        from forge.services.user_paths import user_dir

        base = user_dir(root, user_id, "sandbox")
    else:
        base = root / "sandbox"
    base.mkdir(parents=True, exist_ok=True)

    wrapped = textwrap.dedent(f"""
        import json, sys, math, re, statistics, datetime, collections, itertools
        _SAFE_BUILTINS = {{
            "print": print, "len": len, "range": range, "enumerate": enumerate,
            "zip": zip, "map": map, "filter": filter, "sorted": sorted, "sum": sum,
            "min": min, "max": max, "abs": abs, "round": round, "str": str, "int": int,
            "float": float, "bool": bool, "list": list, "dict": dict, "set": set, "tuple": tuple,
            "True": True, "False": False, "None": None,
        }}
        _stdout = []
        def _print(*a, **k):
            _stdout.append(" ".join(str(x) for x in a))
        _SAFE_BUILTINS["print"] = _print
        # Inject allowlisted modules — user ``import math`` has no __import__.
        _g = {{
            "__builtins__": _SAFE_BUILTINS,
            "json": json, "math": math, "re": re, "statistics": statistics,
            "datetime": datetime, "collections": collections, "itertools": itertools,
            "sys": sys,
        }}
        try:
            exec({code!r}, _g, _g)
        except Exception as e:
            print(json.dumps({{"error": str(e)}}))
        else:
            out = "\\n".join(_stdout)[:{_MAX_OUTPUT}]
            print(out or json.dumps({{"status": "ok"}}))
        """)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", dir=base, delete=False
    ) as f:
        f.write(wrapped)
        script = Path(f.name)
    with contextlib.suppress(OSError):
        os.chmod(script, 0o600)

    run_kwargs: dict = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        "cwd": str(base),
        "env": {
            "PYTHONPATH": "",
            "PATH": os.environ.get("PATH", ""),
            "HOME": str(base),
            "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
        },
        "start_new_session": True,
    }
    if os.name == "posix":
        run_kwargs["preexec_fn"] = subprocess_limits
        run_kwargs["env"]["PATH"] = "/usr/bin:/bin"

    py_args = [sys.executable, "-I", "-S"]
    if sys.version_info >= (3, 11):
        py_args.append("-P")
    py_args.append(str(script))

    job_handle = windows_job_limits()
    try:
        proc = subprocess.Popen(py_args, **run_kwargs)
        assign_windows_job(proc, job_handle)
        try:
            stdout, stderr = proc.communicate(timeout=_TIMEOUT_SEC)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            return json.dumps({"error": f"Timeout after {_TIMEOUT_SEC}s"})
        out = (stdout or stderr or "").strip()[:_MAX_OUTPUT]
        return json.dumps({"stdout": out, "exit_code": proc.returncode})
    finally:
        if job_handle and os.name == "nt":
            try:
                import ctypes

                ctypes.windll.kernel32.CloseHandle(job_handle)  # type: ignore[attr-defined]
            except (AttributeError, OSError):
                pass
        script.unlink(missing_ok=True)
