"""Tests for AST-based code execution sandbox."""

import json

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
    code = (
        f'sys.modules["builtins"].open({str(probe)!r},"w").write("pwn")\n'
        'print("done")'
    )
    result = json.loads(execute_code(code, sandbox_root=str(tmp_path), user_id="u1"))
    assert "error" in result or result.get("exit_code", 0) != 0 or "Name blocked" in str(
        result
    )
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
