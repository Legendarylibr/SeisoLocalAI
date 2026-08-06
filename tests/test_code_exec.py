"""Tests for AST-based code execution sandbox."""

import json
import time

from forge.tools.code_exec import _validate_code, execute_code


def test_blocks_os_import():
    err = _validate_code("import os\nos.listdir('.')")
    assert err is not None
    assert "os" in err


def test_blocks_pathlib_import():
    err = _validate_code("import pathlib\npathlib.Path('/etc/passwd').read_text()")
    assert err is not None
    assert "pathlib" in err


def test_blocks_sys_import():
    err = _validate_code("import sys\nsys.exit(1)")
    assert err is not None


def test_blocks_sys_name_without_import():
    err = _validate_code('sys.modules["builtins"].open("/tmp/x","w")')
    assert err is not None
    assert "sys" in err or "modules" in err


def test_sys_modules_escape_rejected_at_runtime(tmp_path):
    """Regression: injected sys previously allowed host FS writes."""
    probe = tmp_path / "should_not_exist"
    code = f'sys.modules["builtins"].open({str(probe)!r},"w").write("pwn")\nprint("done")'
    result = json.loads(execute_code(code, sandbox_root=str(tmp_path), user_id="u1"))
    assert "error" in result or result.get("exit_code", 0) != 0 or "Name blocked" in str(result)
    assert not probe.exists()


def test_blocks_eval():
    err = _validate_code("eval('1+1')")
    assert err is not None


def test_blocks_gi_frame_escape():
    err = _validate_code("def gen():\n    yield 1\ng = gen()\ng.gi_frame.f_builtins")
    assert err is not None


def test_allows_safe_math():
    result = json.loads(execute_code("print(2 + 2)"))
    assert "4" in result.get("stdout", "")


def test_code_exec_blocks_list_times_large_int():
    from forge.tools.code_exec import _validate_code

    err = _validate_code("x = [0] * 10000000")
    assert err is not None
    assert "too large" in err.lower()


def test_code_exec_blocks_gi_frame():
    err = _validate_code(
        "def f():\n    yield 1\ng = f()\ng.gi_frame.f_builtins['__import__']('os')"
    )
    assert err is not None
    assert "gi_frame" in err or "f_builtins" in err


def test_code_exec_blocks_operator_attrgetter():
    err = _validate_code(
        "import operator\n"
        "cls = operator.attrgetter('__class__', '__bases__')(42)\n"
        "subs = cls.__subclasses__()"
    )
    assert err is not None
    assert "operator" in err or "blocked" in err.lower()


def test_code_exec_blocks_large_list_allocation():
    err = _validate_code("x = [0] * 10**7")
    assert err is not None
    assert "too large" in err.lower()


def test_code_exec_blocks_large_range():
    err = _validate_code("x = list(range(10**7))")
    assert err is not None
    assert "too large" in err.lower()


def test_code_exec_blocks_disallowed_import():
    err = _validate_code("import base64")
    assert err is not None
    assert "blocked" in err.lower()
    err = _validate_code("import _ctypes")
    assert err is not None
    assert "blocked" in err.lower()


def test_code_exec_blocks_underscore_socket():
    err = _validate_code("import _socket")
    assert err is not None
    assert "blocked" in err.lower()


def test_code_exec_child_env_omits_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "should-not-leak")
    monkeypatch.setenv("SEISO_SECRET_KEY", "should-not-leak")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "should-not-leak")
    # Probe via a allowed path: we cannot import os; instead ensure safe math still works
    # while secrets are absent from the scrubbed env builder.
    from forge.tools.code_exec import _scrubbed_child_env

    env = _scrubbed_child_env(tmp_path)
    assert "HF_TOKEN" not in env
    assert "SEISO_SECRET_KEY" not in env
    assert "AWS_SECRET_ACCESS_KEY" not in env
    assert env.get("PYTHONNOUSERSITE") == "1"
    result = json.loads(execute_code("print(1+1)", sandbox_root=str(tmp_path), user_id="u1"))
    assert "2" in result.get("stdout", "")


def test_bignum_pow_rejected_fast():
    start = time.monotonic()
    err = _validate_code("x = 9**9**9**9")
    elapsed = time.monotonic() - start
    assert err is not None
    assert "Allocation too large" in err
    assert elapsed < 1.0


def test_bignum_pow_alloc_rejected_fast():
    start = time.monotonic()
    err = _validate_code("y = [0] * (2**10**10)")
    elapsed = time.monotonic() - start
    assert err is not None
    assert "Allocation too large" in err
    assert elapsed < 1.0


def test_small_powers_allowed():
    assert _validate_code("x = 2**10") is None
    assert _validate_code("y = [0] * 1000") is None


def test_range_checks_all_args():
    err = _validate_code("for i in range(0, 10**10):\n    pass")
    assert err is not None
    assert "Allocation too large" in err


def test_range_small_args_allowed():
    assert _validate_code("for i in range(0, 1000, 2):\n    pass") is None


def test_blocks_ag_frame_escape():
    err = _validate_code("async def f():\n    yield 1\ng = f()\ng.ag_frame")
    assert err is not None
    assert "ag_frame" in err


def test_blocks_cr_code_escape():
    err = _validate_code("async def f():\n    pass\nc = f()\nc.cr_code")
    assert err is not None
    assert "cr_code" in err
