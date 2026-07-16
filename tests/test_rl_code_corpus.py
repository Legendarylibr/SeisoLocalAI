"""Large unit-test-grounded code corpus generator."""

from __future__ import annotations

import json
from pathlib import Path

from seiso.rl_verify.code_corpus import (
    corpus_stats,
    generate_code_corpus,
    generate_grounded_task,
    ground_cases,
    literal_repr,
    parse_mix,
)
from seiso.rl_verify.code_proof import verify_code_proof
from seiso.rl_verify.synth_code import synthesize_code_bundle, validate_task


def test_literal_repr_roundtrip_basics():
    assert literal_repr(True) == "True"
    assert literal_repr([1, "a", None]) == "[1, 'a', None]"
    assert literal_repr({"x": 1}) == "{'x': 1}"


def test_ground_cases_from_executed_solution():
    sol = "def add(a, b):\n    return a + b\n"
    cases = ground_cases(sol, ["add(1, 2)", "add(0, 0)", "add(-1, 4)"])
    assert cases[0] == ("add(1, 2)", "3")
    assert len(cases) >= 3


def test_generate_grounded_task_passes_sandbox():
    for tier in ("easy", "medium", "hard"):
        task = generate_grounded_task(seed=3, index=7, tier=tier)
        validate_task(task)
        assert task.difficulty() == tier
        assert len(task.tests()) >= 2
        result = verify_code_proof(task.completion_for_verifier(), task.sample())
        assert result.passed is True
        # Tests are grounded: solution must not be empty
        assert "def " in task.solution


def test_generate_corpus_scale_and_mix():
    tasks = generate_code_corpus(
        seed=11,
        count=60,
        mix="easy:0.5,medium:0.3,hard:0.2",
        verify=True,
    )
    assert len(tasks) == 60
    stats = corpus_stats(tasks)
    assert stats["by_complexity"]["easy"] >= 20
    assert stats["by_complexity"]["medium"] >= 10
    assert stats["by_complexity"]["hard"] >= 5
    # Diverse families, not a single template.
    families = {t.task_id.split("_")[0] for t in tasks}
    assert len(families) >= 8


def test_corpus_is_deterministic():
    a = generate_code_corpus(seed=5, count=25, verify=True)
    b = generate_code_corpus(seed=5, count=25, verify=True)
    assert [t.task_id for t in a] == [t.task_id for t in b]
    assert [t.tests() for t in a] == [t.tests() for t in b]
    assert [t.solution for t in a] == [t.solution for t in b]


def test_synthesize_bundle_corpus_mode():
    bundle = synthesize_code_bundle(
        seed=0,
        corpus_count=20,
        corpus_mix="easy:0.4,medium:0.4,hard:0.2",
        include_hand_catalog=False,
        build_preferences=True,
        verify=True,
    )
    assert len(bundle.tasks) == 20
    assert len(bundle.preferences) >= 10
    row = bundle.dataset_rows()[0]
    assert row["difficulty"] in {"easy", "medium", "hard"}
    assert row["synth_version"] == 2
    assert row["tests"]


def test_parse_mix_normalizes():
    mix = parse_mix("easy:2,medium:2,hard:1")
    assert abs(sum(mix.values()) - 1.0) < 1e-9
    assert mix["hard"] == 0.2


def test_mini_codebase_tasks_exist_in_hard_stream():
    found = False
    for i in range(80):
        task = generate_grounded_task(seed=0, index=i, tier="hard")
        if "codebase" in task.tags:
            validate_task(task)
            found = True
            # Multi-function solutions look like small modules.
            assert task.solution.count("def ") >= 2
            break
    assert found, "expected at least one mini-codebase hard task"


def test_dataset_row_json_serializable(tmp_path: Path):
    tasks = generate_code_corpus(seed=2, count=5, verify=True)
    path = tmp_path / "out.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(task.to_dataset_row()) + "\n")
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 5
    assert "tests" in json.loads(lines[0])
