"""Deterministic synthetic code tasks with guaranteed unit-test passers.

Design (source-of-truth):
1. Each task is a pure, hand-authored ``solution`` plus I/O cases.
2. ``tests`` are *derived* from those cases (not free-typed separately).
3. Before emit, the solution is run through ``verify_code_proof`` — rows that
   do not pass all tests are rejected (fail closed).
4. Hard negatives are *deterministic mutants* of the solution that must fail
   at least one test. Prefer near-miss mutants over empty/syntax junk.

No LLM is involved; the same seed always yields the same catalog/order.
Use this for slime code datasets, distill preference bootstraps, and CI
guarantees that "a passing completion exists" for every code prompt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from seiso.rl_verify.code_proof import verify_code_proof
from seiso.rl_verify.preferences import (
    ScoredCompletion,
    preference_row_from_pair,
    select_preference_pair,
)

_DEFAULT_TIMEOUT_S = 3.0
_MUTATOR_SEED_SALT = "seiso-synth-code-v1"


@dataclass(frozen=True)
class CodeTask:
    """One verifiable coding problem with a known-good solution."""

    task_id: str
    prompt: str
    solution: str
    cases: tuple[tuple[str, str], ...]
    """Pairs of ``(call_expr, expected_expr)`` used to build asserts.

    Example: ``("add(1, 2)", "3")`` → ``assert add(1, 2) == 3``.
    Use ``is`` for identity checks via expected_expr like ``True`` with
    ``compare="is"``.
    """
    prompt_code: str = ""
    setup: str = ""
    timeout_s: float = _DEFAULT_TIMEOUT_S
    compare: str = "=="
    """``==`` (default) or ``is`` for boolean identity asserts."""
    tags: tuple[str, ...] = ()
    entry_point: str | None = None

    def tests(self) -> list[str]:
        op = "is" if self.compare == "is" else "=="
        return [f"assert {call} {op} {expected}" for call, expected in self.cases]

    def sample(self, *, include_prompt_code: bool = True) -> dict[str, Any]:
        """Verifier sample dict (no solution — used at reward time)."""
        row: dict[str, Any] = {
            "tests": self.tests(),
            "timeout_s": self.timeout_s,
            "benchmark": "code",
        }
        if include_prompt_code and self.prompt_code:
            row["prompt_code"] = self.prompt_code
        if self.setup:
            row["setup"] = self.setup
        if self.entry_point:
            row["entry_point"] = self.entry_point
        return row

    def sample_for_full_program(self) -> dict[str, Any]:
        """Sample for scoring a full fenced program (no double-applied prefix)."""
        return self.sample(include_prompt_code=False)

    def full_source(self) -> str:
        """Complete program text (prompt prefix + solution body when applicable)."""
        parts: list[str] = []
        if self.setup.strip():
            parts.append(self.setup.rstrip())
        if self.prompt_code:
            parts.append(self.prompt_code.rstrip() + "\n" + self.solution.lstrip("\n").rstrip())
        else:
            parts.append(self.solution.rstrip())
        return "\n\n".join(parts) + "\n"

    def fenced_solution(self) -> str:
        return f"```python\n{self.full_source().rstrip()}\n```"

    def completion_for_verifier(self) -> str:
        """Text scored by ``verify_code_proof`` (body-only when prompt_code set)."""
        if self.prompt_code:
            return self.solution
        return self.fenced_solution()

    def difficulty(self) -> str:
        """Complexity tier when tagged (easy|medium|hard), else unknown."""
        for tier in ("easy", "medium", "hard"):
            if tier in self.tags:
                return tier
        return "unknown"

    def to_dataset_row(self) -> dict[str, Any]:
        """JSONL row for slime / distill prompt libraries."""
        row: dict[str, Any] = {
            "prompt_id": self.task_id,
            "prompt": self.prompt,
            "tests": self.tests(),
            # Full program so SFT/distill can train on a known passer; slime reward
            # ignores this field and only scores model completions against tests.
            "solution": self.full_source(),
            "timeout_s": self.timeout_s,
            "benchmark": "code",
            "synth": True,
            "synth_version": 2,
            "difficulty": self.difficulty(),
        }
        if self.prompt_code:
            row["prompt_code"] = self.prompt_code
        if self.setup:
            row["setup"] = self.setup
        if self.entry_point:
            row["entry_point"] = self.entry_point
        if self.tags:
            row["tags"] = list(self.tags)
        return row


@dataclass(frozen=True)
class SyntheticPreference:
    task: CodeTask
    chosen: str
    rejected: str
    pair_kind: str
    chosen_detail: str
    rejected_detail: str

    def to_row(self) -> dict[str, Any]:
        sample = self.task.sample_for_full_program()
        pair = select_preference_pair(
            [
                ScoredCompletion(
                    completion=self.chosen,
                    score=1.0,
                    passed=True,
                    detail=self.chosen_detail,
                    has_code=True,
                ),
                ScoredCompletion(
                    completion=self.rejected,
                    score=0.0,
                    passed=False,
                    detail=self.rejected_detail,
                    has_code=True,
                ),
            ],
            hard_negatives=True,
            require_chosen_pass=True,
        )
        if pair is None:
            # Still emit a deterministic row; scores already known.
            return {
                "prompt_id": self.task.task_id,
                "prompt": self.task.prompt,
                "chosen": self.chosen,
                "rejected": self.rejected,
                "chosen_reward": 1.0,
                "rejected_reward": 0.0,
                "chosen_passed": True,
                "rejected_passed": False,
                "pair_kind": self.pair_kind,
                "reward_source": "synthetic_code_unit_tests",
                "hard_negative": self.pair_kind == "hard_negative",
                "tests": self.task.tests(),
                "benchmark": "code",
                "synth": True,
            }
        return preference_row_from_pair(
            prompt_id=self.task.task_id,
            prompt=self.task.prompt,
            pair=pair,
            sample=sample,
            group_size=2,
            group_rewards=[1.0, 0.0],
            reward_source="synthetic_code_unit_tests",
        )


Mutator = Callable[[str], str | None]


def _mut_swap_add_sub(src: str) -> str | None:
    if " + " in src:
        return src.replace(" + ", " - ", 1)
    if " - " in src:
        return src.replace(" - ", " + ", 1)
    return None


def _mut_swap_mul_div(src: str) -> str | None:
    if " * " in src:
        return src.replace(" * ", " // ", 1)
    if " // " in src:
        return src.replace(" // ", " * ", 1)
    return None


def _mut_invert_compare(src: str) -> str | None:
    for a, b in (("==", "!="), ("!=", "=="), ("<=", ">"), (">=", "<"), ("<", ">="), (">", "<=")):
        if a in src:
            return src.replace(a, b, 1)
    return None


def _mut_not_return_bool(src: str) -> str | None:
    if "return not " in src:
        return src.replace("return not ", "return ", 1)
    if "return " in src and ("%" in src or "==" in src or "is " in src):
        return src.replace("return ", "return not ", 1)
    return None


def _mut_off_by_one(src: str) -> str | None:
    for token, repl in ((" + 1", " + 0"), (" - 1", " - 0"), ("range(1,", "range(0,"), ("range(n)", "range(n-1)")):
        if token in src:
            return src.replace(token, repl, 1)
    return None


def _mut_return_zero(src: str) -> str | None:
    lines = src.splitlines()
    for i in range(len(lines) - 1, -1, -1):
        stripped = lines[i].lstrip()
        if stripped.startswith("return ") and stripped != "return 0":
            indent = lines[i][: len(lines[i]) - len(stripped)]
            lines[i] = f"{indent}return 0"
            return "\n".join(lines) + ("\n" if src.endswith("\n") else "")
    return None


def _mut_return_empty(src: str) -> str | None:
    lines = src.splitlines()
    for i in range(len(lines) - 1, -1, -1):
        stripped = lines[i].lstrip()
        if stripped.startswith("return ") and '""' not in stripped and "[]" not in stripped:
            indent = lines[i][: len(lines[i]) - len(stripped)]
            if "str" in src or "'" in src or '"' in src:
                lines[i] = f'{indent}return ""'
            else:
                lines[i] = f"{indent}return []"
            return "\n".join(lines) + ("\n" if src.endswith("\n") else "")
    return None


_DEFAULT_MUTATORS: tuple[Mutator, ...] = (
    _mut_swap_add_sub,
    _mut_swap_mul_div,
    _mut_invert_compare,
    _mut_not_return_bool,
    _mut_off_by_one,
    _mut_return_zero,
    _mut_return_empty,
)


def _stable_rng(*parts: str) -> random.Random:
    digest = hashlib.sha256(
        (_MUTATOR_SEED_SALT + "|" + "|".join(parts)).encode()
    ).hexdigest()
    return random.Random(int(digest[:16], 16))  # nosec B311 — deterministic catalog


def validate_task(task: CodeTask) -> None:
    """Raise if the canonical solution does not pass all unit tests."""
    result = verify_code_proof(task.completion_for_verifier(), task.sample())
    if not result.passed:
        raise ValueError(
            f"synthetic task {task.task_id!r} solution failed verifier: "
            f"{result.detail} stderr={result.stderr!r}"
        )


def validate_catalog(tasks: Sequence[CodeTask]) -> list[CodeTask]:
    """Return tasks after fail-closed verification (same order)."""
    verified: list[CodeTask] = []
    for task in tasks:
        validate_task(task)
        verified.append(task)
    return verified


def mutate_solution(solution: str, *, task_id: str, seed: int = 0) -> list[str]:
    """Deterministic ordered list of mutant source strings (may be empty)."""
    rng = _stable_rng(task_id, str(seed), solution)
    order = list(_DEFAULT_MUTATORS)
    rng.shuffle(order)
    mutants: list[str] = []
    seen = {solution.strip()}
    for mut in order:
        candidate = mut(solution)
        if candidate is None:
            continue
        key = candidate.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        mutants.append(candidate if candidate.endswith("\n") else candidate + "\n")
    return mutants


def first_failing_mutant(task: CodeTask, *, seed: int = 0) -> str | None:
    """Return fenced mutant that fails ≥1 test, or None if no mutator works."""
    sample = task.sample()
    for mutant_body in mutate_solution(task.solution, task_id=task.task_id, seed=seed):
        # Score the same shape the verifier expects (body-only when prefix exists).
        if task.prompt_code:
            scored = mutant_body
            display = (
                "```python\n"
                + task.prompt_code.rstrip()
                + "\n"
                + mutant_body.lstrip("\n").rstrip()
                + "\n```"
            )
        else:
            scored = f"```python\n{mutant_body.rstrip()}\n```"
            display = scored
        result = verify_code_proof(scored, sample)
        if not result.passed and result.tests_total > 0:
            return display
    return None


def build_preference(task: CodeTask, *, seed: int = 0) -> SyntheticPreference | None:
    """Golden passer + deterministic hard-negative mutant."""
    validate_task(task)
    rejected = first_failing_mutant(task, seed=seed)
    if rejected is None:
        return None
    return SyntheticPreference(
        task=task,
        chosen=task.fenced_solution(),
        rejected=rejected,
        pair_kind="hard_negative",
        chosen_detail="synth:golden_pass",
        rejected_detail="synth:mutant_fail",
    )


# ---------------------------------------------------------------------------
# Fixed catalog (deterministic pure functions — no network, no LLM)
# ---------------------------------------------------------------------------


def _base_catalog() -> list[CodeTask]:
    """Hand-authored tasks; order is part of the deterministic contract."""
    return [
        CodeTask(
            task_id="add",
            prompt="Write a Python function `add(a, b)` that returns the sum of two numbers.",
            solution="def add(a, b):\n    return a + b\n",
            cases=(("add(1, 2)", "3"), ("add(0, 0)", "0"), ("add(-1, 1)", "0")),
            tags=("arith",),
        ),
        CodeTask(
            task_id="sub",
            prompt="Write a Python function `sub(a, b)` that returns a minus b.",
            solution="def sub(a, b):\n    return a - b\n",
            cases=(("sub(5, 2)", "3"), ("sub(0, 0)", "0"), ("sub(1, 4)", "-3")),
            tags=("arith",),
        ),
        CodeTask(
            task_id="mul",
            prompt="Write a Python function `mul(a, b)` that returns the product of two numbers.",
            solution="def mul(a, b):\n    return a * b\n",
            cases=(("mul(3, 4)", "12"), ("mul(0, 9)", "0"), ("mul(-2, 5)", "-10")),
            tags=("arith",),
        ),
        CodeTask(
            task_id="is_even",
            prompt="Write a Python function `is_even(n)` that returns True when n is even.",
            solution="def is_even(n):\n    return n % 2 == 0\n",
            cases=(
                ("is_even(2)", "True"),
                ("is_even(3)", "False"),
                ("is_even(0)", "True"),
            ),
            compare="is",
            tags=("bool",),
        ),
        CodeTask(
            task_id="is_odd",
            prompt="Write a Python function `is_odd(n)` that returns True when n is odd.",
            solution="def is_odd(n):\n    return n % 2 != 0\n",
            cases=(
                ("is_odd(1)", "True"),
                ("is_odd(2)", "False"),
                ("is_odd(-3)", "True"),
            ),
            compare="is",
            tags=("bool",),
        ),
        CodeTask(
            task_id="factorial",
            prompt="Write a Python function `factorial(n)` for non-negative integers (0! = 1).",
            solution=(
                "def factorial(n):\n"
                "    if n < 0:\n"
                "        raise ValueError('n must be non-negative')\n"
                "    out = 1\n"
                "    for i in range(1, n + 1):\n"
                "        out *= i\n"
                "    return out\n"
            ),
            cases=(
                ("factorial(0)", "1"),
                ("factorial(1)", "1"),
                ("factorial(5)", "120"),
            ),
            tags=("math",),
        ),
        CodeTask(
            task_id="reverse_string",
            prompt="Complete the function so it reverses a string.",
            prompt_code="def reverse_string(s: str) -> str:\n",
            solution='    return s[::-1]\n',
            cases=(
                ("reverse_string('ab')", "'ba'"),
                ("reverse_string('')", "''"),
                ("reverse_string('xyz')", "'zyx'"),
            ),
            tags=("string", "prefix"),
        ),
        CodeTask(
            task_id="max_of_two",
            prompt="Write a Python function `max_of_two(a, b)` that returns the larger of two numbers.",
            solution="def max_of_two(a, b):\n    return a if a >= b else b\n",
            cases=(
                ("max_of_two(3, 5)", "5"),
                ("max_of_two(10, 2)", "10"),
                ("max_of_two(-1, -3)", "-1"),
            ),
            tags=("arith",),
        ),
        CodeTask(
            task_id="min_of_two",
            prompt="Write a Python function `min_of_two(a, b)` that returns the smaller of two numbers.",
            solution="def min_of_two(a, b):\n    return a if a <= b else b\n",
            cases=(
                ("min_of_two(3, 5)", "3"),
                ("min_of_two(10, 2)", "2"),
                ("min_of_two(-1, -3)", "-3"),
            ),
            tags=("arith",),
        ),
        CodeTask(
            task_id="count_vowels",
            prompt="Write a Python function `count_vowels(s)` that counts aeiou (case-insensitive).",
            solution=(
                "def count_vowels(s):\n"
                "    vowels = set('aeiouAEIOU')\n"
                "    return sum(1 for ch in s if ch in vowels)\n"
            ),
            cases=(
                ("count_vowels('hello')", "2"),
                ("count_vowels('xyz')", "0"),
                ("count_vowels('AEIOU')", "5"),
            ),
            tags=("string",),
        ),
        CodeTask(
            task_id="is_palindrome",
            prompt="Write a Python function `is_palindrome(s)` that returns True if s equals its reverse.",
            solution="def is_palindrome(s):\n    return s == s[::-1]\n",
            cases=(
                ("is_palindrome('racecar')", "True"),
                ("is_palindrome('hello')", "False"),
                ("is_palindrome('')", "True"),
            ),
            compare="is",
            tags=("string", "bool"),
        ),
        CodeTask(
            task_id="sum_list",
            prompt="Write a Python function `sum_list(nums)` that returns the sum of a list of numbers.",
            solution="def sum_list(nums):\n    total = 0\n    for x in nums:\n        total += x\n    return total\n",
            cases=(
                ("sum_list([1, 2, 3])", "6"),
                ("sum_list([])", "0"),
                ("sum_list([-1, 1])", "0"),
            ),
            tags=("list",),
        ),
        CodeTask(
            task_id="clamp",
            prompt="Write a Python function `clamp(x, lo, hi)` that clamps x into [lo, hi].",
            solution=(
                "def clamp(x, lo, hi):\n"
                "    if x < lo:\n"
                "        return lo\n"
                "    if x > hi:\n"
                "        return hi\n"
                "    return x\n"
            ),
            cases=(
                ("clamp(5, 0, 10)", "5"),
                ("clamp(-1, 0, 10)", "0"),
                ("clamp(99, 0, 10)", "10"),
            ),
            tags=("arith",),
        ),
        CodeTask(
            task_id="last_element",
            prompt="Write a Python function `last_element(items)` that returns the last item of a non-empty list.",
            solution="def last_element(items):\n    return items[-1]\n",
            cases=(
                ("last_element([1, 2, 3])", "3"),
                ("last_element(['a'])", "'a'"),
            ),
            tags=("list",),
        ),
        CodeTask(
            task_id="gcd",
            prompt="Write a Python function `gcd(a, b)` that returns the greatest common divisor of two non-negative integers.",
            solution=(
                "def gcd(a, b):\n"
                "    while b:\n"
                "        a, b = b, a % b\n"
                "    return a\n"
            ),
            cases=(("gcd(12, 8)", "4"), ("gcd(7, 3)", "1"), ("gcd(0, 5)", "5")),
            tags=("math",),
        ),
        CodeTask(
            task_id="double",
            prompt="Complete `double(n)` so it returns twice n.",
            prompt_code="def double(n: int) -> int:\n",
            solution="    return n * 2\n",
            cases=(("double(0)", "0"), ("double(3)", "6"), ("double(-2)", "-4")),
            tags=("arith", "prefix"),
        ),
        CodeTask(
            task_id="unique_sorted",
            prompt="Write a Python function `unique_sorted(nums)` that returns sorted unique integers.",
            solution="def unique_sorted(nums):\n    return sorted(set(nums))\n",
            cases=(
                ("unique_sorted([3, 1, 2, 1])", "[1, 2, 3]"),
                ("unique_sorted([])", "[]"),
                ("unique_sorted([5, 5, 5])", "[5]"),
            ),
            tags=("list",),
        ),
        CodeTask(
            task_id="word_count",
            prompt="Write a Python function `word_count(text)` that counts whitespace-separated words.",
            solution=(
                "def word_count(text):\n"
                "    parts = text.split()\n"
                "    return len(parts)\n"
            ),
            cases=(
                ("word_count('a b c')", "3"),
                ("word_count('')", "0"),
                ("word_count('  hi  there ')", "2"),
            ),
            tags=("string",),
        ),
        CodeTask(
            task_id="fib",
            prompt="Write a Python function `fib(n)` returning the n-th Fibonacci number with fib(0)=0, fib(1)=1.",
            solution=(
                "def fib(n):\n"
                "    if n < 0:\n"
                "        raise ValueError('n must be non-negative')\n"
                "    a, b = 0, 1\n"
                "    for _ in range(n):\n"
                "        a, b = b, a + b\n"
                "    return a\n"
            ),
            cases=(("fib(0)", "0"), ("fib(1)", "1"), ("fib(6)", "8")),
            tags=("math",),
        ),
        CodeTask(
            task_id="is_prime",
            prompt="Write a Python function `is_prime(n)` for integers n >= 0 (0 and 1 are not prime).",
            solution=(
                "def is_prime(n):\n"
                "    if n < 2:\n"
                "        return False\n"
                "    if n % 2 == 0:\n"
                "        return n == 2\n"
                "    d = 3\n"
                "    while d * d <= n:\n"
                "        if n % d == 0:\n"
                "            return False\n"
                "        d += 2\n"
                "    return True\n"
            ),
            cases=(
                ("is_prime(2)", "True"),
                ("is_prime(4)", "False"),
                ("is_prime(1)", "False"),
                ("is_prime(17)", "True"),
            ),
            compare="is",
            tags=("math", "bool"),
        ),
        CodeTask(
            task_id="abs_val",
            prompt="Write a Python function `abs_val(x)` that returns the absolute value of x.",
            solution="def abs_val(x):\n    return x if x >= 0 else -x\n",
            cases=(("abs_val(3)", "3"), ("abs_val(-7)", "7"), ("abs_val(0)", "0")),
            tags=("arith",),
        ),
        CodeTask(
            task_id="mean_ints",
            prompt="Write a Python function `mean_ints(nums)` that returns the integer floor mean of a non-empty list.",
            solution=(
                "def mean_ints(nums):\n"
                "    return sum(nums) // len(nums)\n"
            ),
            cases=(
                ("mean_ints([2, 4, 6])", "4"),
                ("mean_ints([1, 2])", "1"),
                ("mean_ints([10])", "10"),
            ),
            tags=("list", "arith"),
        ),
        CodeTask(
            task_id="starts_with_a",
            prompt="Write a Python function `starts_with_a(s)` that returns True if s starts with 'a' or 'A'.",
            solution=(
                "def starts_with_a(s):\n"
                "    return bool(s) and s[0] in ('a', 'A')\n"
            ),
            cases=(
                ("starts_with_a('apple')", "True"),
                ("starts_with_a('Banana')", "False"),
                ("starts_with_a('')", "False"),
            ),
            compare="is",
            tags=("string", "bool"),
        ),
        CodeTask(
            task_id="square",
            prompt="Write a Python function `square(n)` that returns n squared.",
            solution="def square(n):\n    return n * n\n",
            cases=(("square(0)", "0"), ("square(5)", "25"), ("square(-3)", "9")),
            tags=("arith",),
        ),
    ]


def expand_scaled_variants(
    tasks: Sequence[CodeTask],
    *,
    seed: int = 0,
) -> list[CodeTask]:
    """Deterministic I/O variants for core arithmetic families.

    ``tasks`` is accepted for API symmetry with callers that pass the base
    catalog; expansion is driven by fixed families + ``seed`` only.
    """
    del tasks  # expansion is catalog-driven, not a map over arbitrary tasks
    out: list[CodeTask] = []
    out.extend(_expand_add_family(seed=seed))
    out.extend(_expand_mul_family(seed=seed))
    out.extend(_expand_clamp_family(seed=seed))
    return out


def _unique_case_calls(cases: Sequence[tuple[str, str]]) -> int:
    """Count distinct call expressions (I/O diversity proxy)."""
    return len({call for call, _ in cases})


def _require_diverse_cases(
    task_id: str,
    cases: tuple[tuple[str, str], ...],
    *,
    min_unique: int = 3,
) -> tuple[tuple[str, str], ...]:
    """Fail closed if a synthetic variant lacks distinct I/O triples."""
    if _unique_case_calls(cases) < min_unique:
        raise ValueError(
            f"synthetic variant {task_id!r} has only "
            f"{_unique_case_calls(cases)} unique call(s); need >= {min_unique}"
        )
    return cases


def _expand_add_family(*, seed: int) -> list[CodeTask]:
    rng = _stable_rng("add-family", str(seed))
    # Prefer a != b so commutative pairs still yield two distinct calls.
    pairs = [(1, 2), (10, 20), (7, 8), (100, 1), (-3, 9), (0, 5), (12, 13), (4, 5)]
    rng.shuffle(pairs)
    tasks: list[CodeTask] = []
    for idx, (a, b) in enumerate(pairs):
        # Always include an interior off-diagonal case so near-miss mutants matter.
        c = a + 1
        cases = _require_diverse_cases(
            f"add_v{idx}",
            (
                (f"add({a}, {b})", str(a + b)),
                (f"add({b}, {a})", str(a + b)),
                (f"add({c}, 0)", str(c)),
            ),
        )
        tasks.append(
            CodeTask(
                task_id=f"add_v{idx}",
                prompt="Write a Python function `add(a, b)` that returns the sum of two numbers.",
                solution="def add(a, b):\n    return a + b\n",
                cases=cases,
                tags=("arith", "variant"),
            )
        )
    return tasks


def _expand_mul_family(*, seed: int) -> list[CodeTask]:
    rng = _stable_rng("mul-family", str(seed))
    pairs = [(2, 3), (4, 5), (6, 7), (8, 0), (-2, 4), (9, 1)]
    rng.shuffle(pairs)
    tasks: list[CodeTask] = []
    for idx, (a, b) in enumerate(pairs):
        c = a + 1 if a != 0 else 2
        cases = _require_diverse_cases(
            f"mul_v{idx}",
            (
                (f"mul({a}, {b})", str(a * b)),
                (f"mul({b}, {a})", str(a * b)),
                (f"mul({c}, 1)", str(c)),
            ),
        )
        tasks.append(
            CodeTask(
                task_id=f"mul_v{idx}",
                prompt="Write a Python function `mul(a, b)` that returns the product of two numbers.",
                solution="def mul(a, b):\n    return a * b\n",
                cases=cases,
                tags=("arith", "variant"),
            )
        )
    return tasks


def _expand_clamp_family(*, seed: int) -> list[CodeTask]:
    rng = _stable_rng("clamp-family", str(seed))
    # Each triple spans lo < mid < hi with a distinct interior/exterior probe.
    triples = [
        (5, 0, 10),
        (-1, 0, 10),
        (99, 0, 10),
        (4, 1, 7),
        (7, 1, 5),
        (-5, -3, 3),
    ]
    rng.shuffle(triples)
    tasks: list[CodeTask] = []
    for idx, (x, lo, hi) in enumerate(triples):
        if lo > hi:
            lo, hi = hi, lo
        mid = lo if lo == hi else lo + (hi - lo) // 2
        if mid == lo and hi > lo:
            mid = lo + 1
        expected_x = lo if x < lo else hi if x > hi else x
        # Three distinct probes: primary x, low exterior/edge, high exterior/edge.
        low_probe = lo - 1
        high_probe = hi + 1
        cases = _require_diverse_cases(
            f"clamp_v{idx}",
            (
                (f"clamp({x}, {lo}, {hi})", str(expected_x)),
                (f"clamp({low_probe}, {lo}, {hi})", str(lo)),
                (f"clamp({high_probe}, {lo}, {hi})", str(hi)),
                (f"clamp({mid}, {lo}, {hi})", str(mid)),
            ),
            min_unique=3,
        )
        # Keep three units for scoring density (drop mid if needed for length).
        cases = cases[:3] if _unique_case_calls(cases[:3]) >= 3 else cases
        tasks.append(
            CodeTask(
                task_id=f"clamp_v{idx}",
                prompt="Write a Python function `clamp(x, lo, hi)` that clamps x into [lo, hi].",
                solution=(
                    "def clamp(x, lo, hi):\n"
                    "    if x < lo:\n"
                    "        return lo\n"
                    "    if x > hi:\n"
                    "        return hi\n"
                    "    return x\n"
                ),
                cases=cases,
                tags=("arith", "variant"),
            )
        )
    return tasks


@dataclass
class SynthBundle:
    """Verified tasks plus optional synthetic preference pairs."""

    tasks: list[CodeTask] = field(default_factory=list)
    preferences: list[SyntheticPreference] = field(default_factory=list)

    def dataset_rows(self) -> list[dict[str, Any]]:
        return [t.to_dataset_row() for t in self.tasks]

    def preference_rows(self) -> list[dict[str, Any]]:
        return [p.to_row() for p in self.preferences]


def synthesize_code_bundle(
    *,
    seed: int = 0,
    include_variants: bool = True,
    build_preferences: bool = True,
    limit: int | None = None,
    verify: bool = True,
    # Large unit-test-grounded corpus (solution-first, tests from execution).
    corpus_count: int = 0,
    corpus_mix: str | dict[str, float] | None = None,
    include_hand_catalog: bool = True,
) -> SynthBundle:
    """Build code tasks + optional hard-negative preferences.

    Modes (can combine):

    - **Hand catalog** (default): small fail-closed smoke set + I/O variants.
    - **Corpus** (``corpus_count > 0``): large programmatic generator across
      easy/medium/hard families. Golden solutions are executed to **ground**
      unit tests, then re-checked in the sandbox.
    """
    ordered: list[CodeTask] = []

    if include_hand_catalog and corpus_count <= 0:
        # Classic small catalog path (CI smoke / backward compatible).
        tasks = list(_base_catalog())
        if include_variants:
            tasks.extend(expand_scaled_variants(tasks, seed=seed))
        base_ids = {t.task_id for t in _base_catalog()}
        base = [t for t in tasks if t.task_id in base_ids]
        variants = sorted(
            [t for t in tasks if t.task_id not in base_ids],
            key=lambda t: t.task_id,
        )
        base_order = {t.task_id: i for i, t in enumerate(_base_catalog())}
        base.sort(key=lambda t: base_order[t.task_id])
        ordered = base + variants
    elif corpus_count > 0:
        from seiso.rl_verify.code_corpus import generate_code_corpus

        ordered = generate_code_corpus(
            seed=seed,
            count=corpus_count,
            mix=corpus_mix,
            verify=verify,
            include_hand_catalog=include_hand_catalog,
        )
    else:
        ordered = []

    if limit is not None:
        ordered = ordered[: max(0, limit)]
    if verify and ordered and corpus_count <= 0:
        # Corpus path already verified per-task when verify=True.
        ordered = validate_catalog(ordered)

    prefs: list[SyntheticPreference] = []
    if build_preferences:
        for task in ordered:
            pref = build_preference(task, seed=seed)
            if pref is not None:
                prefs.append(pref)
    return SynthBundle(tasks=ordered, preferences=prefs)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def emit_standard_artifacts(
    *,
    data_dir: Path,
    seed: int = 0,
    verify: bool = True,
    corpus_count: int = 0,
    corpus_mix: str | dict[str, float] | None = None,
    include_hand_catalog: bool = True,
    include_variants: bool = True,
    build_preferences: bool = True,
    limit: int | None = None,
    slime_name: str = "slime_code_sample.jsonl",
    distill_name: str = "distill_code_synth.jsonl",
    prefs_name: str = "synthetic_code_preferences.jsonl",
) -> dict[str, Any]:
    """Write slime code sample + distill prompts + synthetic preference JSONL.

    For large coding corpora set ``corpus_count`` (e.g. 2000). Smoke defaults
    keep the small hand catalog when ``corpus_count=0``.
    """
    bundle = synthesize_code_bundle(
        seed=seed,
        include_variants=include_variants,
        build_preferences=build_preferences,
        limit=limit,
        verify=verify,
        corpus_count=corpus_count,
        corpus_mix=corpus_mix,
        include_hand_catalog=include_hand_catalog,
    )
    slime_path = data_dir / slime_name
    distill_code_path = data_dir / distill_name
    pref_path = data_dir / prefs_name

    # Slime rows: prompt/tests/solution (model never sees solution at reward time).
    n_slime = write_jsonl(slime_path, bundle.dataset_rows())
    n_distill = write_jsonl(distill_code_path, bundle.dataset_rows())
    n_pref = write_jsonl(pref_path, bundle.preference_rows())
    stats: dict[str, Any] = {
        "tasks": len(bundle.tasks),
        "preferences": len(bundle.preferences),
        "slime_code_sample": n_slime,
        "distill_code_synth": n_distill,
        "synthetic_code_preferences": n_pref,
        "corpus_count": corpus_count,
        "seed": seed,
    }
    if corpus_count > 0:
        from seiso.rl_verify.code_corpus import corpus_stats

        stats["corpus"] = corpus_stats(bundle.tasks)
    return stats


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Synthesize unit-test-grounded coding tasks (solution-first). "
            "Use --count for large multi-complexity corpora."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory for JSONL artifacts (default: data/)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Catalog / corpus seed")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of tasks after generation",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=0,
        help=(
            "Generate this many corpus tasks across complexity tiers "
            "(0 = small hand catalog + variants only)"
        ),
    )
    parser.add_argument(
        "--mix",
        type=str,
        default="easy:0.4,medium:0.4,hard:0.2",
        help="Complexity mix for --count, e.g. easy:0.3,medium:0.4,hard:0.3",
    )
    parser.add_argument(
        "--no-hand-catalog",
        action="store_true",
        help="When using --count, do not prepend the hand-authored smoke catalog",
    )
    parser.add_argument(
        "--no-preferences",
        action="store_true",
        help="Skip hard-negative preference pair generation",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip sandbox verification (not recommended)",
    )
    parser.add_argument(
        "--stdout-tasks",
        action="store_true",
        help="Print task JSONL to stdout instead of writing standard artifacts",
    )
    parser.add_argument(
        "--slime-name",
        type=str,
        default="slime_code_sample.jsonl",
        help="Output filename for slime prompts (under --data-dir)",
    )
    parser.add_argument(
        "--prefs-name",
        type=str,
        default="synthetic_code_preferences.jsonl",
        help="Output filename for preference pairs",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    corpus_count = max(0, int(args.count))
    include_hand = not args.no_hand_catalog

    if args.stdout_tasks:
        bundle = synthesize_code_bundle(
            seed=args.seed,
            include_variants=corpus_count == 0,
            build_preferences=False,
            limit=args.limit,
            verify=not args.no_verify,
            corpus_count=corpus_count,
            corpus_mix=args.mix,
            include_hand_catalog=include_hand if corpus_count > 0 else True,
        )
        for row in bundle.dataset_rows():
            print(json.dumps(row, ensure_ascii=False, sort_keys=True))
        return 0

    stats = emit_standard_artifacts(
        data_dir=args.data_dir,
        seed=args.seed,
        verify=not args.no_verify,
        corpus_count=corpus_count,
        corpus_mix=args.mix,
        include_hand_catalog=include_hand if corpus_count > 0 else True,
        include_variants=corpus_count == 0,
        build_preferences=not args.no_preferences,
        limit=args.limit,
        slime_name=args.slime_name,
        prefs_name=args.prefs_name,
    )
    print(json.dumps(stats, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
