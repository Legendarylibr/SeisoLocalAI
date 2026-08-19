"""Hard-negative preference pairs from unit-test verification."""

from __future__ import annotations

from seiso.distill_rl.prompts import RolloutPrompt, is_verifiable_prompt, prompt_to_verifier_sample
from seiso.rl_verify.preferences import (
    ScoredCompletion,
    preference_row_from_pair,
    score_code_completion,
    select_preference_pair,
)


def test_select_pair_prefers_pass_over_fail_with_code():
    pass_sol = ScoredCompletion(
        completion="def add(a,b): return a+b",
        score=1.0,
        passed=True,
        has_code=True,
        detail="code:pass 2/2",
    )
    hard_fail = ScoredCompletion(
        completion="def add(a,b): return a-b",
        score=0.0,
        passed=False,
        has_code=True,
        detail="code:pass 0/2",
    )
    empty_fail = ScoredCompletion(
        completion="I cannot solve this",
        score=0.0,
        passed=False,
        has_code=False,
        detail="code:no_extractable_python",
    )
    pair = select_preference_pair(
        [empty_fail, hard_fail, pass_sol],
        hard_negatives=True,
        require_chosen_pass=True,
    )
    assert pair is not None
    assert pair.chosen.completion == pass_sol.completion
    assert pair.rejected.completion == hard_fail.completion
    assert pair.pair_kind == "hard_negative"


def test_select_pair_prefers_near_miss_among_fails():
    chosen = ScoredCompletion(completion="ok", score=1.0, passed=True, has_code=True)
    near = ScoredCompletion(
        completion="near", score=0.5, passed=False, has_code=True, tests_passed=1
    )
    far = ScoredCompletion(completion="far", score=0.0, passed=False, has_code=True, tests_passed=0)
    pair = select_preference_pair([far, near, chosen], hard_negatives=True)
    assert pair is not None
    assert pair.rejected.completion == "near"


def test_select_pair_soft_fail_prefers_shorter_equal_score():
    chosen = ScoredCompletion(completion="ok", score=1.0, passed=True, has_code=False)
    short_wrong = ScoredCompletion(completion="41", score=0.0, passed=False, has_code=False)
    long_wrong = ScoredCompletion(
        completion="definitely not forty one padded " * 3,
        score=0.0,
        passed=False,
        has_code=False,
    )
    pair = select_preference_pair([long_wrong, short_wrong, chosen], hard_negatives=True)
    assert pair is not None
    assert pair.rejected.completion == "41"


def test_select_pair_hard_fail_and_chosen_prefer_shorter_on_ties():
    short_pass = ScoredCompletion(
        completion="def f():\n    return 1\n",
        score=1.0,
        passed=True,
        has_code=True,
    )
    long_pass = ScoredCompletion(
        completion="def f():\n    return 1\n# " + ("pad " * 40),
        score=1.0,
        passed=True,
        has_code=True,
    )
    short_fail = ScoredCompletion(
        completion="def f():\n    return 0\n",
        score=0.0,
        passed=False,
        has_code=True,
        tests_passed=0,
    )
    long_fail = ScoredCompletion(
        completion="def f():\n    return 0\n# " + ("pad " * 40),
        score=0.0,
        passed=False,
        has_code=True,
        tests_passed=0,
    )
    pair = select_preference_pair(
        [long_pass, short_pass, long_fail, short_fail],
        hard_negatives=True,
    )
    assert pair is not None
    assert pair.chosen.completion == short_pass.completion
    assert pair.rejected.completion == short_fail.completion


def test_select_pair_skips_when_no_pass():
    a = ScoredCompletion(completion="a", score=0.0, passed=False, has_code=True)
    b = ScoredCompletion(completion="b", score=0.0, passed=False, has_code=True)
    assert select_preference_pair([a, b], require_chosen_pass=True) is None


def test_select_pair_skips_when_all_pass_equal():
    a = ScoredCompletion(completion="a", score=1.0, passed=True, has_code=True)
    b = ScoredCompletion(completion="b", score=1.0, passed=True, has_code=True)
    assert select_preference_pair([a, b], require_chosen_pass=True) is None


def test_score_and_row_roundtrip():
    good = "def add(a, b):\n    return a + b\n"
    bad = "def add(a, b):\n    return a - b\n"
    sample = {"tests": ["assert add(1, 2) == 3"]}
    scored = [
        score_code_completion(good, sample),
        score_code_completion(bad, sample),
    ]
    pair = select_preference_pair(scored, hard_negatives=True)
    assert pair is not None
    assert pair.chosen.passed is True
    assert pair.rejected.passed is False
    row = preference_row_from_pair(
        prompt_id="p1",
        prompt="write add",
        pair=pair,
        sample=sample,
        group_size=2,
        group_rewards=[c.score for c in scored],
    )
    assert row["hard_negative"] is True
    assert row["reward_source"] == "code_unit_tests"
    assert row["chosen_reward"] > row["rejected_reward"]


def test_prompt_code_fields_are_verifiable():
    prompt = RolloutPrompt(
        prompt_id="c1",
        text="Write add",
        tests=["assert add(1,1)==2"],
        benchmark="code",
    )
    assert is_verifiable_prompt(prompt)
    sample = prompt_to_verifier_sample(prompt)
    assert sample["tests"] == ["assert add(1,1)==2"]
