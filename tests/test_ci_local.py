"""Regression tests for local CI selection and diagnostic baselines."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts import run_ci_local


def test_baseline_ignores_location_changes(tmp_path):
    baseline = tmp_path / "baseline.txt"
    baseline.write_text(
        "seiso/example.py:10:2: error: Bad value  [arg-type]\n",
        encoding="utf-8",
    )

    run_ci_local._baseline_check(
        label="mypy",
        baseline_path=baseline,
        current_lines=["seiso/example.py:99:8: error: Bad value  [arg-type]"],
        update_baseline=False,
    )


def test_baseline_preserves_duplicate_counts(tmp_path):
    baseline = tmp_path / "baseline.txt"
    baseline.write_text(
        "seiso/example.py:10: error: Bad value  [arg-type]\n",
        encoding="utf-8",
    )

    with pytest.raises(subprocess.CalledProcessError):
        run_ci_local._baseline_check(
            label="mypy",
            baseline_path=baseline,
            current_lines=[
                "seiso/example.py:20: error: Bad value  [arg-type]",
                "seiso/example.py:30: error: Bad value  [arg-type]",
            ],
            update_baseline=False,
        )


def test_baseline_normalizes_windows_path_separators(tmp_path):
    baseline = tmp_path / "baseline.txt"
    baseline.write_text(
        "C:\\repo\\seiso\\example.py:10: error: Bad value  [arg-type]\n",
        encoding="utf-8",
    )

    run_ci_local._baseline_check(
        label="mypy",
        baseline_path=baseline,
        current_lines=["C:/repo/seiso/example.py:40: error: Bad value  [arg-type]"],
        update_baseline=False,
    )


def test_types_fails_when_mypy_crashes(monkeypatch, tmp_path):
    responses = iter(
        [
            subprocess.CompletedProcess([], 0, stdout="3.10\n", stderr=""),
            subprocess.CompletedProcess([], 2, stdout="", stderr="mypy: internal error\n"),
        ]
    )
    monkeypatch.setattr(
        run_ci_local.subprocess,
        "run",
        lambda *_args, **_kwargs: next(responses),
    )

    with pytest.raises(subprocess.CalledProcessError):
        run_ci_local.job_types(
            tmp_path,
            "python",
            {},
            update_baseline=False,
        )


def test_changed_source_runs_full_test_suite(monkeypatch, tmp_path):
    calls: list[list[str]] = []
    monkeypatch.setattr(
        run_ci_local,
        "_step",
        lambda _title, command, **_kwargs: calls.append(list(command)),
    )

    run_ci_local.job_test(
        tmp_path,
        "python",
        {},
        files=[Path("seiso/inference/runner.py")],
        workers=0,
        hardware_tests=False,
    )

    assert "tests/" in calls[0]


def test_changed_test_only_runs_direct_test(monkeypatch, tmp_path):
    calls: list[list[str]] = []
    monkeypatch.setattr(
        run_ci_local,
        "_step",
        lambda _title, command, **_kwargs: calls.append(list(command)),
    )

    run_ci_local.job_test(
        tmp_path,
        "python",
        {},
        files=[Path("tests/test_runner.py")],
        workers=0,
        hardware_tests=False,
    )

    assert "tests/test_runner.py" in calls[0]
    assert "tests/" not in calls[0]


def test_changed_docs_skip_tests(monkeypatch, tmp_path):
    calls: list[list[str]] = []
    monkeypatch.setattr(
        run_ci_local,
        "_step",
        lambda _title, command, **_kwargs: calls.append(list(command)),
    )

    run_ci_local.job_test(
        tmp_path,
        "python",
        {},
        files=[Path("docs/CI_LOCAL.md")],
        workers=0,
        hardware_tests=False,
    )

    assert calls == []


def test_pytest_worker_args_support_auto_and_worksteal():
    assert run_ci_local._pytest_worker_args(0, "loadscope") == []
    assert run_ci_local._pytest_worker_args("auto", "worksteal") == [
        "-n",
        "auto",
        "--dist",
        "worksteal",
    ]
    assert run_ci_local._pytest_worker_args(4, "loadfile") == [
        "-n",
        "4",
        "--dist",
        "loadfile",
    ]
