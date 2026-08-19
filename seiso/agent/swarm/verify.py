"""Deterministic completion/correctness checks, optional tiny JSON judge."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from seiso.agent.swarm.types import Verdict

JudgeFn = Callable[[str, str, Mapping[str, Any]], str]

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_PY_PATH_RE = re.compile(r"(?:^|[\s`\"'(])((?:[\w./-]+)\.py)\b")
_MAX_COMPILE_FILES = 20


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item).strip()]
    return []


def extract_workdir_py_files(text: str, workdir: Path) -> list[Path]:
    """Paths mentioned in harness output that exist under *workdir* (capped)."""
    root = workdir.resolve()
    found: list[Path] = []
    seen: set[Path] = set()
    for match in _PY_PATH_RE.finditer(text or ""):
        raw = match.group(1)
        path = Path(raw)
        if not path.is_absolute():
            path = root / raw
        try:
            resolved = path.resolve()
            resolved.relative_to(root)
        except (OSError, ValueError):
            continue
        if not resolved.is_file() or resolved in seen:
            continue
        seen.add(resolved)
        found.append(resolved)
        if len(found) >= _MAX_COMPILE_FILES:
            break
    return found


def compile_python_files(paths: list[Path]) -> tuple[bool, list[str]]:
    """Byte-compile mentioned files. Cheap correctness signal, no extra model."""
    import py_compile

    errors: list[str] = []
    for path in paths:
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            errors.append(str(exc)[:200])
        except OSError as exc:
            errors.append(f"{path.name}: {exc}")
    return not errors, errors


def collect_repo_signals(output: Mapping[str, Any], *, workdir: Path | None) -> dict[str, Any]:
    if workdir is None:
        return {"compile_ok": True, "checked_files": [], "compile_errors": []}
    text = "\n".join(str(output.get(key) or "") for key in ("stdout", "stderr", "detail", "goal"))
    files = extract_workdir_py_files(text, workdir)
    if not files:
        return {"compile_ok": True, "checked_files": [], "compile_errors": []}
    ok, errors = compile_python_files(files)
    return {
        "compile_ok": ok,
        "checked_files": [str(path) for path in files],
        "compile_errors": errors,
    }


def check_completion(output: Mapping[str, Any], *, workdir: Path | None = None) -> Verdict:
    evidence: dict[str, Any] = {}
    if output.get("dry_run"):
        return Verdict(True, "dry_run", {"dry_run": True})
    if output.get("blocked") == "oom_guard":
        return Verdict(False, "oom_guard", dict(output))
    exit_code = output.get("exit_code")
    evidence["exit_code"] = exit_code
    if exit_code not in (None, 0):
        return Verdict(False, f"exit:{exit_code}", evidence)
    missing: list[str] = []
    root = Path(workdir) if workdir is not None else None
    for rel in _as_list(output.get("expect_files") or output.get("artifacts")):
        path = Path(rel)
        if root is not None and not path.is_absolute():
            path = root / rel
        if not path.exists():
            missing.append(rel)
    if missing:
        evidence["missing"] = missing
        return Verdict(False, "missing_artifacts", evidence)
    status = str(output.get("status") or "")
    if status in {"failed", "blocked"}:
        return Verdict(False, status, evidence)
    return Verdict(True, "complete", evidence)


def check_correctness(
    output: Mapping[str, Any],
    *,
    workdir: Path | None = None,
    tests_ran: bool | None = None,
    tests_ok: bool | None = None,
) -> Verdict:
    evidence: dict[str, Any] = {}
    if output.get("dry_run"):
        return Verdict(True, "dry_run", {"dry_run": True})
    if tests_ran is None:
        tests_ran = bool(output.get("tests_ran"))
    if tests_ok is None:
        tests_ok = output.get("tests_ok")
    evidence["tests_ran"] = tests_ran
    evidence["tests_ok"] = tests_ok
    if tests_ran and tests_ok is False:
        return Verdict(False, "tests_failed", evidence)
    if tests_ran and tests_ok is True:
        return Verdict(True, "tests_passed", evidence)
    signals = collect_repo_signals(output, workdir=workdir)
    evidence.update(signals)
    if signals.get("compile_ok") is False:
        return Verdict(False, "compile_failed", evidence)
    completion = check_completion(output, workdir=workdir)
    if not completion.ok:
        return Verdict(False, completion.reason, {**evidence, **dict(completion.evidence)})
    return Verdict(True, "inconclusive_no_tests", evidence)


def parse_judge_json(raw: str) -> Verdict:
    text = (raw or "").strip()
    if not text:
        return Verdict(False, "judge_empty", used_llm=True)
    blob = text
    match = _JSON_RE.search(text)
    if match:
        blob = match.group(0)
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        return Verdict(False, "judge_parse_failed", {"raw": text[:240]}, used_llm=True)
    if not isinstance(data, dict):
        return Verdict(False, "judge_not_object", used_llm=True)
    ok = bool(data.get("ok"))
    reason = str(data.get("reason") or data.get("reasons") or ("ok" if ok else "judge_rejected"))
    if isinstance(data.get("reasons"), list):
        reason = "; ".join(str(item) for item in data["reasons"][:4]) or reason
    return Verdict(ok, reason[:240], {"judge": data}, used_llm=True)


DEFAULT_JUDGE_PROMPT = (
    "You are a strict verifier. Reply with JSON only: "
    '{"ok": true|false, "confidence": 0-1, "reasons": ["..."]}. '
    "Do not claim ok if tests failed or required files are missing."
)


def maybe_judge(
    verdict: Verdict,
    *,
    allow_llm: bool,
    system_prompt: str,
    evidence: Mapping[str, Any],
    judge: JudgeFn | None,
    preflight_ok: bool,
) -> Verdict:
    if verdict.ok and verdict.reason != "inconclusive_no_tests":
        return verdict
    if not allow_llm or judge is None:
        return verdict
    if not preflight_ok:
        return Verdict(
            verdict.ok,
            "oom_guard" if not verdict.ok else verdict.reason,
            {**dict(verdict.evidence), "llm_skipped": "oom_guard"},
            used_llm=False,
        )
    prompt = (system_prompt or "").strip() or DEFAULT_JUDGE_PROMPT
    raw = judge(prompt, verdict.reason, evidence)
    judged = parse_judge_json(raw)
    if verdict.reason == "tests_failed":
        return Verdict(False, "tests_failed", {**dict(verdict.evidence), **dict(judged.evidence)})
    return judged
