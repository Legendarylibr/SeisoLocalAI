"""Tests for AST-based code execution sandbox."""

import json

from forge.tools.code_exec import execute_code, _validate_code


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


def test_blocks_eval():
    err = _validate_code("eval('1+1')")
    assert err is not None


def test_blocks_gi_frame_escape():
    err = _validate_code(
        "def gen():\n    yield 1\n"
        "g = gen()\n"
        "g.gi_frame.f_builtins"
    )
    assert err is not None


def test_allows_safe_math():
    result = json.loads(execute_code("print(2 + 2)"))
    assert "4" in result.get("stdout", "")
