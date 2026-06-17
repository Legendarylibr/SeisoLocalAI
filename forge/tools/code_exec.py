"""Sandboxed Python code execution — AST-validated, restricted builtins."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

from seiso.security import resolve_data_dir

from forge.security.audit import audit_event

_BLOCKED_MODULES = frozenset(
    {
        "operator",
        "_operator",
        "functools",
        "_functools",
        "copy",
        "weakref",
        "os",
        "subprocess",
        "socket",
        "shutil",
        "ctypes",
        "multiprocessing",
        "pickle",
        "builtins",
        "importlib",
        "importlib.util",
        "pty",
        "fcntl",
        "resource",
        "signal",
        "pathlib",
        "sys",
        "io",
        "code",
        "gc",
        "inspect",
        "types",
        "glob",
        "tempfile",
        "sqlite3",
        "http",
        "urllib",
        "requests",
        "webbrowser",
        "site",
        "runpy",
        "_io",
        "ssl",
        "ftplib",
        "smtplib",
        "marshal",
        "select",
        "selectors",
        "threading",
        "asyncio",
        "email",
        "zipfile",
        "tarfile",
        "gzip",
        "bz2",
        "lzma",
        "binascii",
        "struct",
        "array",
        "mmap",
        "msvcrt",
        "winreg",
        "winsound",
        "posix",
        "pwd",
        "grp",
        "termios",
        "tty",
        "readline",
        "curses",
        "dbm",
        "shelve",
        "xmlrpc",
        "xml",
        "html",
        "nntplib",
        "poplib",
        "imaplib",
        "telnetlib",
        "aifc",
        "sunau",
        "wave",
        "chunk",
        "colorsys",
        "crypt",
        "hmac",
        "secrets",
        "uuid",
        "venv",
        "ensurepip",
        "pip",
        "setuptools",
        "distutils",
        "configparser",
        "logging",
        "warnings",
        "traceback",
        "linecache",
        "dis",
        "pickletools",
        "doctest",
        "unittest",
        "pdb",
        "profile",
        "pstats",
        "timeit",
        "trace",
        "tracemalloc",
        "faulthandler",
        "atexit",
        "symtable",
        "tokenize",
        "keyword",
        "token",
        "ast",
        "parser",
        "symbol",
        "tabnanny",
        "py_compile",
        "compileall",
        "zipimport",
        "pkgutil",
        "modulefinder",
        "imp",
    }
)
_BLOCKED_NAMES = frozenset({"eval", "exec", "compile", "open", "__import__", "breakpoint", "getattr", "setattr"})
_BLOCKED_ATTRS = frozenset(
    {
        "gi_frame",
        "gi_code",
        "gi_yieldfrom",
        "f_back",
        "f_builtins",
        "f_globals",
        "f_locals",
        "f_code",
        "cr_frame",
        "tb_frame",
        "tb_next",
        "__class__",
        "__bases__",
        "__subclasses__",
        "__mro__",
        "__init__",
        "__globals__",
        "__code__",
        "__builtins__",
        "__dict__",
        "__getattribute__",
        "__reduce__",
        "__reduce_ex__",
    }
)
_TIMEOUT_SEC = 15
_MAX_OUTPUT = 16_000
_MAX_CODE_LEN = 8_000


class _CodeValidator(ast.NodeVisitor):
    def __init__(self) -> None:
        self.errors: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.split(".")[0]
            if root in _BLOCKED_MODULES:
                self.errors.append(f"Import blocked: {alias.name}")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            root = node.module.split(".")[0]
            if root in _BLOCKED_MODULES:
                self.errors.append(f"Import blocked: {node.module}")

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in _BLOCKED_NAMES:
            self.errors.append(f"Call blocked: {node.func.id}()")
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


def execute_code(code: str, sandbox_root: str | None = None) -> str:
    """Run user code in isolated subprocess with AST pre-check."""
    err = _validate_code(code)
    if err:
        return json.dumps({"error": err})

    audit_event("code_exec", code_len=len(code))

    base = Path(sandbox_root) if sandbox_root else resolve_data_dir() / "sandbox"
    base.mkdir(parents=True, exist_ok=True)

    wrapped = textwrap.dedent(
        f"""
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
        _g = {{"__builtins__": _SAFE_BUILTINS}}
        try:
            exec({code!r}, _g, _g)
        except Exception as e:
            print(json.dumps({{"error": str(e)}}))
        else:
            out = "\\n".join(_stdout)[:{_MAX_OUTPUT}]
            print(out or json.dumps({{"status": "ok"}}))
        """
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", dir=base, delete=False) as f:
        f.write(wrapped)
        script = Path(f.name)

    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-S", str(script)],
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_SEC,
            cwd=str(base),
            env={"PYTHONPATH": "", "PATH": "/usr/bin:/bin", "HOME": str(base)},
        )
        out = (proc.stdout or proc.stderr or "").strip()[:_MAX_OUTPUT]
        return json.dumps({"stdout": out, "exit_code": proc.returncode})
    except subprocess.TimeoutExpired:
        return json.dumps({"error": f"Timeout after {_TIMEOUT_SEC}s"})
    finally:
        script.unlink(missing_ok=True)
