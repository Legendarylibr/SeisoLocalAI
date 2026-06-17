"""Sandboxed Python code execution — AST-validated, restricted builtins."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

from forge.security.audit import audit_event
from seiso.security import resolve_data_dir

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
_BLOCKED_NAMES = frozenset({
    "eval", "exec", "compile", "open", "__import__", "breakpoint", "getattr", "setattr",
    "delattr", "hasattr", "type", "object", "classmethod", "staticmethod", "super",
    "vars", "input", "globals", "locals", "memoryview", "bytearray", "bytes",
    "help", "dir", "property", "__build_class__",
})
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
_MAX_RSS_BYTES = 256 * 1024 * 1024
_MAX_OPEN_FDS = 32


def _subprocess_limits() -> None:
    """Apply resource limits in the code-exec child process (Unix only)."""
    import resource

    try:
        resource.setrlimit(resource.RLIMIT_CPU, (_TIMEOUT_SEC, _TIMEOUT_SEC + 5))
        resource.setrlimit(resource.RLIMIT_AS, (_MAX_RSS_BYTES, _MAX_RSS_BYTES))
        if hasattr(resource, "RLIMIT_NOFILE"):
            resource.setrlimit(resource.RLIMIT_NOFILE, (_MAX_OPEN_FDS, _MAX_OPEN_FDS))
    except (OSError, ValueError):
        pass


def _windows_job_limits() -> int | None:
    """Return a Windows job handle with process memory cap, or None if unavailable."""
    if os.name != "nt":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            return None

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_ulonglong),
                ("WriteOperationCount", ctypes.c_ulonglong),
                ("OtherOperationCount", ctypes.c_ulonglong),
                ("ReadTransferCount", ctypes.c_ulonglong),
                ("WriteTransferCount", ctypes.c_ulonglong),
                ("OtherTransferCount", ctypes.c_ulonglong),
            ]

        class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", wintypes.LARGE_INTEGER),
                ("PerJobUserTimeLimit", wintypes.LARGE_INTEGER),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = 0x100  # JOB_OBJECT_LIMIT_PROCESS_MEMORY
        info.ProcessMemoryLimit = _MAX_RSS_BYTES
        JobObjectExtendedLimitInformation = 9
        kernel32.SetInformationJobObject(
            job,
            JobObjectExtendedLimitInformation,
            ctypes.byref(info),
            ctypes.sizeof(info),
        )
        return int(job)
    except (AttributeError, OSError, ValueError):
        return None


def _assign_windows_job(proc: subprocess.Popen[str], job_handle: int | None) -> None:
    if os.name != "nt" or not job_handle:
        return
    try:
        import ctypes

        ctypes.windll.kernel32.AssignProcessToJobObject(  # type: ignore[attr-defined]
            job_handle,
            int(proc._handle),  # noqa: SLF001
        )
    except (AttributeError, OSError, ValueError):
        pass


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

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load) and node.id in _BLOCKED_NAMES:
            self.errors.append(f"Name blocked: {node.id}")
        self.generic_visit(node)

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


def execute_code(code: str, sandbox_root: str | None = None, user_id: str | None = None) -> str:
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
        os.chmod(script, 0o600)
    except OSError:
        pass

    run_kwargs: dict = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        "cwd": str(base),
        "env": {
            "PYTHONPATH": "",
            "PATH": os.environ.get("PATH", ""),
            "HOME": str(base),
            "SystemRoot": os.environ.get("SystemRoot", ""),
        },
        "start_new_session": True,
    }
    if os.name == "posix":
        run_kwargs["preexec_fn"] = _subprocess_limits
        run_kwargs["env"]["PATH"] = "/usr/bin:/bin"

    py_args = [sys.executable, "-I", "-S"]
    if sys.version_info >= (3, 11):
        py_args.append("-P")
    py_args.append(str(script))

    job_handle = _windows_job_limits()
    try:
        proc = subprocess.Popen(py_args, **run_kwargs)
        _assign_windows_job(proc, job_handle)
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
