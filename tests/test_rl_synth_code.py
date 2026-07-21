"""Deterministic synthetic code tasks: passers + hard-negative mutants."""

from __future__ import annotations

import json
from pathlib import Path

from seiso.rl_verify.code_proof import verify_code_proof
from seiso.rl_verify.synth_code import (
    build_preference,
    emit_held_out_eval_jsonl,
    emit_standard_artifacts,
    synthesize_code_bundle,
    validate_task,
)


def test_base_catalog_solutions_all_pass():
    bundle = synthesize_code_bundle(
        seed=0,
        include_variants=False,
        build_preferences=False,
        verify=True,
    )
    assert len(bundle.tasks) >= 16
    for task in bundle.tasks:
        result = verify_code_proof(task.completion_for_verifier(), task.sample())
        assert result.passed, (task.task_id, result.detail, result.stderr)


def test_synthesize_is_deterministic():
    a = synthesize_code_bundle(seed=7, build_preferences=True, verify=True)
    b = synthesize_code_bundle(seed=7, build_preferences=True, verify=True)
    assert [t.task_id for t in a.tasks] == [t.task_id for t in b.tasks]
    assert [t.solution for t in a.tasks] == [t.solution for t in b.tasks]
    assert [t.tests() for t in a.tasks] == [t.tests() for t in b.tasks]
    assert [p.to_row() for p in a.preferences] == [p.to_row() for p in b.preferences]


def test_different_seeds_change_variant_case_order():
    a = synthesize_code_bundle(seed=0, include_variants=True, build_preferences=False)
    b = synthesize_code_bundle(seed=1, include_variants=True, build_preferences=False)
    cases_a = [t.cases for t in a.tasks if "_v" in t.task_id]
    cases_b = [t.cases for t in b.tasks if "_v" in t.task_id]
    assert cases_a
    assert cases_a != cases_b


def test_variants_have_diverse_io_cases():
    bundle = synthesize_code_bundle(
        seed=0, include_variants=True, build_preferences=False, verify=True
    )
    variants = [t for t in bundle.tasks if "_v" in t.task_id]
    assert variants
    for task in variants:
        calls = {call for call, _ in task.cases}
        assert len(calls) >= 3, (task.task_id, task.cases)
        assert len(task.cases) >= 3


def test_preference_hard_negative_fails_verifier():
    bundle = synthesize_code_bundle(
        seed=0,
        include_variants=False,
        build_preferences=True,
        limit=8,
        verify=True,
    )
    assert bundle.preferences
    for pref in bundle.preferences:
        sample = pref.task.sample_for_full_program()
        good = verify_code_proof(pref.chosen, sample)
        bad = verify_code_proof(pref.rejected, sample)
        assert good.passed is True
        assert bad.passed is False
        row = pref.to_row()
        assert row["chosen_passed"] is True
        assert row["rejected_passed"] is False
        assert row["reward_source"] == "synthetic_code_unit_tests"
        assert row["hard_negative"] is True


def test_dataset_row_includes_solution_and_derived_tests():
    bundle = synthesize_code_bundle(
        seed=0, include_variants=False, build_preferences=False, limit=1
    )
    row = bundle.dataset_rows()[0]
    assert "solution" in row
    assert row["tests"]
    assert row["synth"] is True
    validate_task(bundle.tasks[0])


def test_emit_standard_artifacts(tmp_path: Path):
    stats = emit_standard_artifacts(data_dir=tmp_path, seed=0, verify=True)
    assert stats["tasks"] >= 16
    assert stats["preferences"] >= 8
    slime = (tmp_path / "slime_code_sample.jsonl").read_text(encoding="utf-8").strip()
    prefs = (tmp_path / "synthetic_code_preferences.jsonl").read_text(encoding="utf-8").strip()
    assert slime
    assert prefs
    first = json.loads(slime.splitlines()[0])
    assert "tests" in first and "solution" in first
    # Golden full solution must pass (do not double-apply prompt_code).
    sample = {"tests": first["tests"], "timeout_s": first.get("timeout_s", 3)}
    assert verify_code_proof(
        f"```python\n{first['solution']}```",
        sample,
    ).passed


def test_build_preference_none_when_no_mutant(monkeypatch):
    from seiso.rl_verify import synth_code as mod

    bundle = synthesize_code_bundle(
        seed=0, include_variants=False, build_preferences=False, limit=1
    )
    task = bundle.tasks[0]
    monkeypatch.setattr(mod, "mutate_solution", lambda *a, **k: [])
    assert build_preference(task, seed=0) is None


def test_distill_synthetic_code_preference_bundle(tmp_path: Path):
    from seiso.distill_rl.preferences import build_synthetic_code_preference_bundle

    out = build_synthetic_code_preference_bundle(
        output_dir=tmp_path / "prefs",
        seed=0,
        train_fraction=0.75,
        limit=12,
        include_variants=False,
    )
    assert out.train_count >= 1
    assert out.val_count >= 1
    assert out.train_path.is_file()
    train_line = out.train_path.read_text(encoding="utf-8").splitlines()[0]
    row = json.loads(train_line)
    assert row["chosen_passed"] is True
    assert row["rejected_passed"] is False
    assert row["reward_source"] == "synthetic_code_unit_tests"


def test_emit_held_out_eval_disjoint_from_train(tmp_path: Path):
    train = synthesize_code_bundle(
        seed=0,
        include_variants=False,
        build_preferences=False,
        verify=True,
        corpus_count=8,
        include_hand_catalog=False,
    )
    train_ids = {str(t.task_id) for t in train.tasks}
    n, eval_seed = emit_held_out_eval_jsonl(
        data_dir=tmp_path,
        count=8,
        seed=0,
        train_seed=0,
        verify=True,
        train_prompt_ids=train_ids,
    )
    assert n == 8
    assert eval_seed == 10_007
    rows = [
        json.loads(line)
        for line in (tmp_path / "slime_code_eval.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len(rows) == 8
    for row in rows:
        assert row["held_out"] is True
        assert str(row["prompt_id"]).startswith("eval_")
        bare = str(row["prompt_id"]).removeprefix("eval_")
        assert bare not in train_ids
        assert verify_code_proof(
            f"```python\n{row['solution']}```",
            {"tests": row["tests"], "timeout_s": row.get("timeout_s", 3)},
        ).passed


def test_emit_standard_artifacts_with_eval(tmp_path: Path):
    stats = emit_standard_artifacts(
        data_dir=tmp_path,
        seed=0,
        verify=True,
        corpus_count=4,
        include_hand_catalog=False,
        include_variants=False,
        build_preferences=False,
        eval_count=4,
    )
    assert stats["slime_code_eval"] == 4
    assert stats["eval_seed"] == 10_007
    assert (tmp_path / "slime_code_eval.jsonl").is_file()


def test_distill_synthetic_default_train_fraction_holds_out_val(tmp_path: Path):
    from seiso.distill_rl.preferences import build_synthetic_code_preference_bundle

    out = build_synthetic_code_preference_bundle(
        output_dir=tmp_path / "prefs",
        seed=1,
        limit=40,
        include_variants=False,
    )
    assert out.train_count > out.val_count >= 1
    frac = out.train_count / (out.train_count + out.val_count)
    assert 0.8 <= frac <= 0.9
