"""Large-scale unit-test-grounded coding corpus for improving base model coding.

Unlike the small hand catalog in ``synth_code``, this module **generates** many
tasks programmatically:

1. Sample a family + complexity tier + parameters (deterministic seed).
2. Render a pure-Python golden solution (single function or mini-codebase).
3. Sample diverse call expressions (inputs).
4. **Ground tests** by executing the golden solution in-process to get expected
   outputs, then emit ``assert call == expected`` lines.
5. Fail-closed re-check via the sandboxed ``verify_code_proof``.

No LLM is used. Same ``seed`` + ``count`` + mix ⇒ same corpus.
"""

from __future__ import annotations

import ast
import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from seiso.rl_verify.synth_code import (
    CodeTask,
    _stable_rng,
    validate_task,
)

# ---------------------------------------------------------------------------
# Safe execution of *our own* generated goldens (not model completions)
# ---------------------------------------------------------------------------

_SAFE_BUILTINS: dict[str, Any] = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "pow": pow,
    "range": range,
    "reversed": reversed,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
    "True": True,
    "False": False,
    "None": None,
}


def literal_repr(value: Any) -> str:
    """Stable Python literal for assert RHS (ints/bools/strs/lists/dicts)."""
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        return "None"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise ValueError(f"non-finite float not allowed: {value!r}")
        # Prefer ints when exact.
        if value.is_integer() and abs(value) < 10**12:
            return str(int(value))
        return repr(value)
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(literal_repr(v) for v in value) + "]"
    if isinstance(value, tuple):
        if len(value) == 1:
            return "(" + literal_repr(value[0]) + ",)"
        return "(" + ", ".join(literal_repr(v) for v in value) + ")"
    if isinstance(value, dict):
        items = ", ".join(
            f"{literal_repr(k)}: {literal_repr(v)}" for k, v in value.items()
        )
        return "{" + items + "}"
    if isinstance(value, set):
        if not value:
            return "set()"
        return "{" + ", ".join(literal_repr(v) for v in sorted(value, key=repr)) + "}"
    raise TypeError(f"unsupported expected type: {type(value)!r}")


def exec_golden_call(solution: str, call_expr: str) -> Any:
    """Run a generated golden solution and evaluate one call expression."""
    # Reject obviously dangerous AST shapes before exec.
    tree = ast.parse(solution)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal)):
            raise ValueError("imports/globals not allowed in synthetic goldens")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValueError("dunder attribute access not allowed")
    # Single namespace: functions defined under exec(..., globals, locals) with
    # distinct dicts store defs in locals but resolve free names via globals,
    # so multi-def mini-modules would NameError on helpers.
    ns: dict[str, Any] = {"__builtins__": _SAFE_BUILTINS}
    exec(solution, ns)  # nosec B102 — own goldens
    return eval(call_expr, ns)  # nosec B307


def ground_cases(
    solution: str,
    call_exprs: Sequence[str],
    *,
    compare: str = "==",
) -> tuple[tuple[str, str], ...]:
    """Execute golden on each call; return (call, expected_literal) pairs.

    ``compare`` selects the assert operator later (``==`` vs ``is``). Expected
    values are always stored via ``literal_repr`` so bool/None identity asserts
    and value equality asserts share the same grounding path.
    """
    del compare  # operator is applied when emitting asserts, not when grounding.
    cases: list[tuple[str, str]] = []
    seen: set[str] = set()
    for call in call_exprs:
        call = call.strip()
        if not call or call in seen:
            continue
        expected = exec_golden_call(solution, call)
        cases.append((call, literal_repr(expected)))
        seen.add(call)
    if len(cases) < 2:
        raise ValueError(f"need >=2 grounded cases, got {len(cases)}")
    return tuple(cases)


# ---------------------------------------------------------------------------
# Input samplers
# ---------------------------------------------------------------------------


def _rand_int(rng: random.Random, lo: int = -20, hi: int = 40) -> int:
    return rng.randint(lo, hi)


def _rand_list(rng: random.Random, *, n: int | None = None, lo: int = -9, hi: int = 20) -> list[int]:
    length = n if n is not None else rng.randint(0, 6)
    return [_rand_int(rng, lo, hi) for _ in range(length)]


def _rand_str(rng: random.Random, *, max_len: int = 8) -> str:
    alphabet = "abcdefghijklmnopqrstuvwxyz "
    n = rng.randint(0, max_len)
    return "".join(rng.choice(alphabet) for _ in range(n)).strip() or rng.choice("abcdef")


def _list_lit(xs: Sequence[Any]) -> str:
    return literal_repr(list(xs))


def _str_lit(s: str) -> str:
    return literal_repr(s)


# ---------------------------------------------------------------------------
# Family registry: each yields (prompt, solution, call_exprs, compare, tags)
# ---------------------------------------------------------------------------

FamilyFn = Callable[[random.Random, int], tuple[str, str, list[str], str, tuple[str, ...]]]


@dataclass(frozen=True)
class FamilySpec:
    name: str
    complexity: str  # easy | medium | hard
    weight: float
    generate: FamilyFn


def _f_binop(rng: random.Random, idx: int) -> tuple[str, str, list[str], str, tuple[str, ...]]:
    op, name, py = rng.choice(
        [
            ("+", "add", "a + b"),
            ("-", "sub", "a - b"),
            ("*", "mul", "a * b"),
            ("//", "floordiv", "a // b if b != 0 else 0"),
            ("%", "mod", "a % b if b != 0 else 0"),
        ]
    )
    del op
    fname = f"{name}_g{idx}"
    sol = f"def {fname}(a, b):\n    return {py}\n"
    calls: list[str] = []
    for _ in range(5):
        a, b = _rand_int(rng), _rand_int(rng, -10, 15)
        if "b != 0" in py and b == 0:
            b = rng.choice([-3, -1, 1, 2, 4])
        calls.append(f"{fname}({a}, {b})")
    # Distinct second args.
    calls.append(f"{fname}(0, 1)")
    prompt = (
        f"Write a pure Python function `{fname}(a, b)` implementing integer "
        f"binary op `{py}` (return 0 when dividing/modding by zero)."
    )
    return prompt, sol, calls, "==", ("arith", "easy", "generated")


def _f_unary(rng: random.Random, idx: int) -> tuple[str, str, list[str], str, tuple[str, ...]]:
    kind = rng.choice(["abs", "square", "negate", "double", "inc", "dec"])
    fname = f"{kind}_g{idx}"
    body = {
        "abs": "x if x >= 0 else -x",
        "square": "x * x",
        "negate": "-x",
        "double": "x * 2",
        "inc": "x + 1",
        "dec": "x - 1",
    }[kind]
    sol = f"def {fname}(x):\n    return {body}\n"
    calls = [f"{fname}({_rand_int(rng, -15, 15)})" for _ in range(5)]
    prompt = f"Write Python `{fname}(x)` that returns ({body})."
    return prompt, sol, calls, "==", ("arith", "easy", "generated")


def _f_predicate(rng: random.Random, idx: int) -> tuple[str, str, list[str], str, tuple[str, ...]]:
    kind = rng.choice(["is_even", "is_odd", "is_positive", "is_nonneg", "is_zero"])
    fname = f"{kind}_g{idx}"
    body = {
        "is_even": "n % 2 == 0",
        "is_odd": "n % 2 != 0",
        "is_positive": "n > 0",
        "is_nonneg": "n >= 0",
        "is_zero": "n == 0",
    }[kind]
    sol = f"def {fname}(n):\n    return {body}\n"
    calls = [f"{fname}({_rand_int(rng, -8, 12)})" for _ in range(5)]
    prompt = f"Write `{fname}(n)` returning True when ({body})."
    return prompt, sol, calls, "is", ("bool", "easy", "generated")


def _f_string_easy(rng: random.Random, idx: int) -> tuple[str, str, list[str], str, tuple[str, ...]]:
    kind = rng.choice(["strlen", "reverse", "upper", "lower", "first_char"])
    fname = f"{kind}_g{idx}"
    bodies = {
        "strlen": "len(s)",
        "reverse": "s[::-1]",
        "upper": "s.upper()",
        "lower": "s.lower()",
        "first_char": "s[0] if s else ''",
    }
    sol = f"def {fname}(s):\n    return {bodies[kind]}\n"
    calls = [f"{fname}({_str_lit(_rand_str(rng))})" for _ in range(4)]
    calls.append(f"{fname}({_str_lit('')})")
    prompt = f"Write `{fname}(s)` for strings: return {bodies[kind]}."
    return prompt, sol, calls, "==", ("string", "easy", "generated")


def _f_list_easy(rng: random.Random, idx: int) -> tuple[str, str, list[str], str, tuple[str, ...]]:
    kind = rng.choice(["sum_list", "len_list", "product_or_one", "max_or_none", "min_or_none"])
    fname = f"{kind}_g{idx}"
    if kind == "sum_list":
        body = "sum(nums)"
        sol = f"def {fname}(nums):\n    return {body}\n"
    elif kind == "len_list":
        body = "len(nums)"
        sol = f"def {fname}(nums):\n    return {body}\n"
    elif kind == "product_or_one":
        sol = (
            f"def {fname}(nums):\n"
            f"    out = 1\n"
            f"    for x in nums:\n"
            f"        out *= x\n"
            f"    return out\n"
        )
        body = "product of nums (1 if empty)"
    elif kind == "max_or_none":
        sol = f"def {fname}(nums):\n    return max(nums) if nums else None\n"
        body = "max or None"
    else:
        sol = f"def {fname}(nums):\n    return min(nums) if nums else None\n"
        body = "min or None"
    calls = [f"{fname}({_list_lit(_rand_list(rng))})" for _ in range(4)]
    calls.append(f"{fname}([])")
    prompt = f"Write `{fname}(nums)` over int lists: {body}."
    return prompt, sol, calls, "==", ("list", "easy", "generated")


def _f_clamp_gen(rng: random.Random, idx: int) -> tuple[str, str, list[str], str, tuple[str, ...]]:
    fname = f"clamp_g{idx}"
    sol = (
        f"def {fname}(x, lo, hi):\n"
        f"    if lo > hi:\n"
        f"        lo, hi = hi, lo\n"
        f"    if x < lo:\n"
        f"        return lo\n"
        f"    if x > hi:\n"
        f"        return hi\n"
        f"    return x\n"
    )
    calls: list[str] = []
    for _ in range(5):
        lo, hi = sorted([_rand_int(rng, -5, 10), _rand_int(rng, 0, 20)])
        x = _rand_int(rng, -10, 25)
        calls.append(f"{fname}({x}, {lo}, {hi})")
    prompt = f"Write `{fname}(x, lo, hi)` clamping x into [lo, hi] (swap if lo>hi)."
    return prompt, sol, calls, "==", ("arith", "easy", "generated")


def _f_filter_map(rng: random.Random, idx: int) -> tuple[str, str, list[str], str, tuple[str, ...]]:
    kind = rng.choice(["evens", "odds", "positives", "squares", "double_all"])
    fname = f"{kind}_g{idx}"
    if kind == "evens":
        sol = f"def {fname}(nums):\n    return [x for x in nums if x % 2 == 0]\n"
    elif kind == "odds":
        sol = f"def {fname}(nums):\n    return [x for x in nums if x % 2 != 0]\n"
    elif kind == "positives":
        sol = f"def {fname}(nums):\n    return [x for x in nums if x > 0]\n"
    elif kind == "squares":
        sol = f"def {fname}(nums):\n    return [x * x for x in nums]\n"
    else:
        sol = f"def {fname}(nums):\n    return [x * 2 for x in nums]\n"
    calls = [f"{fname}({_list_lit(_rand_list(rng, n=rng.randint(1, 7)))})" for _ in range(4)]
    calls.append(f"{fname}([])")
    prompt = f"Write `{fname}(nums)` returning a new list for pattern `{kind}`."
    return prompt, sol, calls, "==", ("list", "medium", "generated")


def _f_string_medium(rng: random.Random, idx: int) -> tuple[str, str, list[str], str, tuple[str, ...]]:
    kind = rng.choice(
        ["count_vowels", "is_palindrome", "word_count", "strip_spaces", "title_words"]
    )
    fname = f"{kind}_g{idx}"
    if kind == "count_vowels":
        sol = (
            f"def {fname}(s):\n"
            f"    vowels = set('aeiouAEIOU')\n"
            f"    return sum(1 for ch in s if ch in vowels)\n"
        )
    elif kind == "is_palindrome":
        sol = f"def {fname}(s):\n    return s == s[::-1]\n"
        calls = [f"{fname}({_str_lit(s)})" for s in ["abba", "abc", "", "a", "racecar", "no"]]
        prompt = f"Write `{fname}(s)` True iff s is a palindrome."
        return prompt, sol, calls, "is", ("string", "medium", "generated")
    elif kind == "word_count":
        sol = f"def {fname}(text):\n    return len(text.split())\n"
    elif kind == "strip_spaces":
        sol = f"def {fname}(s):\n    return ''.join(s.split())\n"
    else:
        sol = (
            f"def {fname}(s):\n"
            f"    return ' '.join(w[:1].upper() + w[1:].lower() for w in s.split() if w)\n"
        )
    calls = [f"{fname}({_str_lit(_rand_str(rng, max_len=12))})" for _ in range(4)]
    calls.append(f"{fname}({_str_lit('  hi  there ')})")
    prompt = f"Write `{fname}` implementing `{kind}` on strings."
    return prompt, sol, calls, "==", ("string", "medium", "generated")


def _f_dict_count(rng: random.Random, idx: int) -> tuple[str, str, list[str], str, tuple[str, ...]]:
    fname = f"char_count_g{idx}"
    sol = (
        f"def {fname}(s):\n"
        f"    out = {{}}\n"
        f"    for ch in s:\n"
        f"        out[ch] = out.get(ch, 0) + 1\n"
        f"    return out\n"
    )
    samples = ["aab", "xyz", "", "mississippi", "Aa", _rand_str(rng, max_len=6)]
    calls = [f"{fname}({_str_lit(s)})" for s in samples]
    prompt = f"Write `{fname}(s)` returning a dict of character counts."
    return prompt, sol, calls, "==", ("dict", "medium", "generated")


def _f_unique_order(rng: random.Random, idx: int) -> tuple[str, str, list[str], str, tuple[str, ...]]:
    fname = f"unique_preserve_g{idx}"
    sol = (
        f"def {fname}(items):\n"
        f"    seen = set()\n"
        f"    out = []\n"
        f"    for x in items:\n"
        f"        if x not in seen:\n"
        f"            seen.add(x)\n"
        f"            out.append(x)\n"
        f"    return out\n"
    )
    calls = []
    for _ in range(4):
        xs = _rand_list(rng, n=rng.randint(2, 8), lo=0, hi=6)
        calls.append(f"{fname}({_list_lit(xs)})")
    calls.append(f"{fname}([])")
    prompt = f"Write `{fname}(items)` unique values preserving first-seen order."
    return prompt, sol, calls, "==", ("list", "medium", "generated")


def _f_running_sum(rng: random.Random, idx: int) -> tuple[str, str, list[str], str, tuple[str, ...]]:
    fname = f"running_sum_g{idx}"
    sol = (
        f"def {fname}(nums):\n"
        f"    out = []\n"
        f"    total = 0\n"
        f"    for x in nums:\n"
        f"        total += x\n"
        f"        out.append(total)\n"
        f"    return out\n"
    )
    calls = [f"{fname}({_list_lit(_rand_list(rng, n=rng.randint(1, 6)))})" for _ in range(4)]
    calls.append(f"{fname}([])")
    prompt = f"Write `{fname}(nums)` returning prefix sums."
    return prompt, sol, calls, "==", ("list", "medium", "generated")


def _f_gcd_lcm(rng: random.Random, idx: int) -> tuple[str, str, list[str], str, tuple[str, ...]]:
    kind = rng.choice(["gcd", "lcm"])
    fname = f"{kind}_g{idx}"
    if kind == "gcd":
        sol = (
            f"def {fname}(a, b):\n"
            f"    a, b = abs(a), abs(b)\n"
            f"    while b:\n"
            f"        a, b = b, a % b\n"
            f"    return a\n"
        )
    else:
        sol = (
            f"def {fname}(a, b):\n"
            f"    def _gcd(x, y):\n"
            f"        while y:\n"
            f"            x, y = y, x % y\n"
            f"        return x\n"
            f"    a, b = abs(a), abs(b)\n"
            f"    if a == 0 or b == 0:\n"
            f"        return 0\n"
            f"    return a // _gcd(a, b) * b\n"
        )
    calls = []
    for _ in range(5):
        a, b = abs(_rand_int(rng, 0, 30)), abs(_rand_int(rng, 0, 30))
        calls.append(f"{fname}({a}, {b})")
    prompt = f"Write `{fname}(a, b)` computing non-negative {kind}."
    return prompt, sol, calls, "==", ("math", "medium", "generated")


def _f_anagram(rng: random.Random, idx: int) -> tuple[str, str, list[str], str, tuple[str, ...]]:
    fname = f"is_anagram_g{idx}"
    sol = (
        f"def {fname}(a, b):\n"
        f"    return sorted(a.replace(' ', '').lower()) == sorted(b.replace(' ', '').lower())\n"
    )
    pairs = [
        ("listen", "silent"),
        ("hello", "world"),
        ("", ""),
        ("Dormitory", "dirty room"),
        ("abc", "ab"),
        (_rand_str(rng, max_len=5), _rand_str(rng, max_len=5)),
    ]
    calls = [f"{fname}({_str_lit(a)}, {_str_lit(b)})" for a, b in pairs]
    prompt = f"Write `{fname}(a, b)` True if anagrams ignoring spaces/case."
    return prompt, sol, calls, "is", ("string", "medium", "generated")


def _f_parens(rng: random.Random, idx: int) -> tuple[str, str, list[str], str, tuple[str, ...]]:
    fname = f"valid_parens_g{idx}"
    sol = (
        f"def {fname}(s):\n"
        f"    depth = 0\n"
        f"    for ch in s:\n"
        f"        if ch == '(':\n"
        f"            depth += 1\n"
        f"        elif ch == ')':\n"
        f"            depth -= 1\n"
        f"            if depth < 0:\n"
        f"                return False\n"
        f"    return depth == 0\n"
    )
    samples = ["()", "(())", "(()", ")(", "", "()()", "())(", "(()())"]
    calls = [f"{fname}({_str_lit(s)})" for s in samples]
    prompt = f"Write `{fname}(s)` True iff parentheses in s are balanced."
    return prompt, sol, calls, "is", ("string", "stack", "medium", "generated")


def _f_two_sum(rng: random.Random, idx: int) -> tuple[str, str, list[str], str, tuple[str, ...]]:
    """Return indices of two numbers summing to target (or empty list)."""
    fname = f"two_sum_g{idx}"
    sol = (
        f"def {fname}(nums, target):\n"
        f"    seen = {{}}\n"
        f"    for i, x in enumerate(nums):\n"
        f"        need = target - x\n"
        f"        if need in seen:\n"
        f"            return [seen[need], i]\n"
        f"        seen[x] = i\n"
        f"    return []\n"
    )
    calls: list[str] = []
    for _ in range(4):
        xs = _rand_list(rng, n=rng.randint(3, 7), lo=-5, hi=15)
        if len(xs) >= 2 and rng.random() < 0.7:
            i, j = 0, len(xs) - 1
            target = xs[i] + xs[j]
        else:
            target = _rand_int(rng, -5, 20)
        calls.append(f"{fname}({_list_lit(xs)}, {target})")
    calls.append(f"{fname}([1, 2, 3], 100)")
    prompt = (
        f"Write `{fname}(nums, target)` returning [i, j] of first pair summing "
        f"to target (left-to-right), or [] if none."
    )
    return prompt, sol, calls, "==", ("list", "algo", "hard", "generated")


def _f_binary_search(rng: random.Random, idx: int) -> tuple[str, str, list[str], str, tuple[str, ...]]:
    fname = f"binary_search_g{idx}"
    sol = (
        f"def {fname}(nums, target):\n"
        f"    lo, hi = 0, len(nums) - 1\n"
        f"    while lo <= hi:\n"
        f"        mid = (lo + hi) // 2\n"
        f"        if nums[mid] == target:\n"
        f"            return mid\n"
        f"        if nums[mid] < target:\n"
        f"            lo = mid + 1\n"
        f"        else:\n"
        f"            hi = mid - 1\n"
        f"    return -1\n"
    )
    calls: list[str] = []
    for _ in range(4):
        xs = sorted(set(_rand_list(rng, n=rng.randint(3, 8), lo=0, hi=30)))
        target = rng.choice(xs) if xs and rng.random() < 0.6 else _rand_int(rng, 0, 30)
        calls.append(f"{fname}({_list_lit(xs)}, {target})")
    calls.append(f"{fname}([], 1)")
    prompt = f"Write `{fname}(nums, target)` binary search on sorted nums; -1 if missing."
    return prompt, sol, calls, "==", ("algo", "hard", "generated")


def _f_merge_sorted(rng: random.Random, idx: int) -> tuple[str, str, list[str], str, tuple[str, ...]]:
    fname = f"merge_sorted_g{idx}"
    sol = (
        f"def {fname}(a, b):\n"
        f"    i = j = 0\n"
        f"    out = []\n"
        f"    while i < len(a) and j < len(b):\n"
        f"        if a[i] <= b[j]:\n"
        f"            out.append(a[i])\n"
        f"            i += 1\n"
        f"        else:\n"
        f"            out.append(b[j])\n"
        f"            j += 1\n"
        f"    out.extend(a[i:])\n"
        f"    out.extend(b[j:])\n"
        f"    return out\n"
    )
    calls = []
    for _ in range(4):
        a = sorted(_rand_list(rng, n=rng.randint(0, 5), lo=0, hi=15))
        b = sorted(_rand_list(rng, n=rng.randint(0, 5), lo=0, hi=15))
        calls.append(f"{fname}({_list_lit(a)}, {_list_lit(b)})")
    prompt = f"Write `{fname}(a, b)` merging two sorted ascending int lists."
    return prompt, sol, calls, "==", ("algo", "list", "hard", "generated")


def _f_mini_codebase_stats(rng: random.Random, idx: int) -> tuple[str, str, list[str], str, tuple[str, ...]]:
    """Multi-function mini module: helpers + public API (codebase-like)."""
    api = f"summarize_g{idx}"
    sol = (
        f"def _mean_{idx}(nums):\n"
        f"    if not nums:\n"
        f"        return 0\n"
        f"    return sum(nums) / len(nums)\n"
        f"\n"
        f"def _clamp_{idx}(x, lo, hi):\n"
        f"    if x < lo:\n"
        f"        return lo\n"
        f"    if x > hi:\n"
        f"        return hi\n"
        f"    return x\n"
        f"\n"
        f"def {api}(nums, lo=0, hi=100):\n"
        f"    \"\"\"Return dict with count, mean (clamped into [lo,hi]), and max.\"\"\"\n"
        f"    if not nums:\n"
        f"        return {{'count': 0, 'mean': 0, 'max': None}}\n"
        f"    m = _mean_{idx}(nums)\n"
        f"    return {{\n"
        f"        'count': len(nums),\n"
        f"        'mean': _clamp_{idx}(m, lo, hi),\n"
        f"        'max': max(nums),\n"
        f"    }}\n"
    )
    calls = []
    for _ in range(4):
        xs = _rand_list(rng, n=rng.randint(0, 6), lo=-5, hi=40)
        lo, hi = 0, 20
        calls.append(f"{api}({_list_lit(xs)}, {lo}, {hi})")
    calls.append(f"{api}([])")
    prompt = (
        f"Implement a small module with helpers and public `{api}(nums, lo=0, hi=100)` "
        f"returning {{'count', 'mean', 'max'}} where mean is clamped into [lo, hi]. "
        f"Empty input → count 0, mean 0, max None. Pure Python only."
    )
    return prompt, sol, calls, "==", ("codebase", "dict", "hard", "generated")


def _f_mini_codebase_text(rng: random.Random, idx: int) -> tuple[str, str, list[str], str, tuple[str, ...]]:
    api = f"normalize_tokens_g{idx}"
    sol = (
        f"def _split_words_{idx}(text):\n"
        f"    return [w for w in text.lower().split() if w]\n"
        f"\n"
        f"def _dedupe_{idx}(words):\n"
        f"    seen = set()\n"
        f"    out = []\n"
        f"    for w in words:\n"
        f"        if w not in seen:\n"
        f"            seen.add(w)\n"
        f"            out.append(w)\n"
        f"    return out\n"
        f"\n"
        f"def {api}(text, unique=True):\n"
        f"    words = _split_words_{idx}(text)\n"
        f"    if unique:\n"
        f"        words = _dedupe_{idx}(words)\n"
        f"    return words\n"
    )
    texts = [
        "Hello Hello World",
        "  A b A  ",
        "",
        "Python PYTHON python",
        _rand_str(rng, max_len=20),
    ]
    calls = []
    for t in texts:
        calls.append(f"{api}({_str_lit(t)}, True)")
        calls.append(f"{api}({_str_lit(t)}, False)")
    prompt = (
        f"Implement helpers plus `{api}(text, unique=True)` that lowercases, splits "
        f"on whitespace, and optionally de-duplicates preserving order."
    )
    return prompt, sol, calls, "==", ("codebase", "string", "hard", "generated")


def _f_matrix_flatten(rng: random.Random, idx: int) -> tuple[str, str, list[str], str, tuple[str, ...]]:
    fname = f"flatten_g{idx}"
    sol = (
        f"def {fname}(matrix):\n"
        f"    out = []\n"
        f"    for row in matrix:\n"
        f"        for x in row:\n"
        f"            out.append(x)\n"
        f"    return out\n"
    )
    calls = []
    for _ in range(4):
        rows = rng.randint(0, 3)
        matrix = [_rand_list(rng, n=rng.randint(0, 4), lo=0, hi=9) for _ in range(rows)]
        calls.append(f"{fname}({literal_repr(matrix)})")
    calls.append(f"{fname}([])")
    prompt = f"Write `{fname}(matrix)` row-major flatten of list[list[int]]."
    return prompt, sol, calls, "==", ("list", "hard", "generated")


def _f_rotate(rng: random.Random, idx: int) -> tuple[str, str, list[str], str, tuple[str, ...]]:
    fname = f"rotate_left_g{idx}"
    sol = (
        f"def {fname}(items, k):\n"
        f"    if not items:\n"
        f"        return []\n"
        f"    n = len(items)\n"
        f"    k = k % n\n"
        f"    return items[k:] + items[:k]\n"
    )
    calls = []
    for _ in range(4):
        xs = _rand_list(rng, n=rng.randint(1, 6), lo=0, hi=9)
        k = rng.randint(0, 10)
        calls.append(f"{fname}({_list_lit(xs)}, {k})")
    calls.append(f"{fname}([], 3)")
    prompt = f"Write `{fname}(items, k)` left-rotate list by k (mod len)."
    return prompt, sol, calls, "==", ("list", "medium", "generated")


def _f_digit_ops(rng: random.Random, idx: int) -> tuple[str, str, list[str], str, tuple[str, ...]]:
    kind = rng.choice(["digit_sum", "digit_count", "reverse_digits"])
    fname = f"{kind}_g{idx}"
    if kind == "digit_sum":
        sol = (
            f"def {fname}(n):\n"
            f"    n = abs(n)\n"
            f"    total = 0\n"
            f"    while n:\n"
            f"        total += n % 10\n"
            f"        n //= 10\n"
            f"    return total\n"
        )
    elif kind == "digit_count":
        sol = (
            f"def {fname}(n):\n"
            f"    n = abs(n)\n"
            f"    if n == 0:\n"
            f"        return 1\n"
            f"    c = 0\n"
            f"    while n:\n"
            f"        c += 1\n"
            f"        n //= 10\n"
            f"    return c\n"
        )
    else:
        sol = (
            f"def {fname}(n):\n"
            f"    sign = -1 if n < 0 else 1\n"
            f"    n = abs(n)\n"
            f"    rev = 0\n"
            f"    while n:\n"
            f"        rev = rev * 10 + n % 10\n"
            f"        n //= 10\n"
            f"    return sign * rev\n"
        )
    calls = [f"{fname}({_rand_int(rng, -9999, 9999)})" for _ in range(5)]
    calls.append(f"{fname}(0)")
    prompt = f"Write `{fname}(n)` for integers implementing `{kind}`."
    return prompt, sol, calls, "==", ("math", "medium", "generated")


FAMILIES: tuple[FamilySpec, ...] = (
    # easy
    FamilySpec("binop", "easy", 1.2, _f_binop),
    FamilySpec("unary", "easy", 1.0, _f_unary),
    FamilySpec("predicate", "easy", 1.0, _f_predicate),
    FamilySpec("string_easy", "easy", 1.0, _f_string_easy),
    FamilySpec("list_easy", "easy", 1.0, _f_list_easy),
    FamilySpec("clamp", "easy", 0.8, _f_clamp_gen),
    # medium
    FamilySpec("filter_map", "medium", 1.2, _f_filter_map),
    FamilySpec("string_medium", "medium", 1.1, _f_string_medium),
    FamilySpec("dict_count", "medium", 0.9, _f_dict_count),
    FamilySpec("unique_order", "medium", 1.0, _f_unique_order),
    FamilySpec("running_sum", "medium", 0.9, _f_running_sum),
    FamilySpec("gcd_lcm", "medium", 0.9, _f_gcd_lcm),
    FamilySpec("anagram", "medium", 0.8, _f_anagram),
    FamilySpec("parens", "medium", 0.8, _f_parens),
    FamilySpec("rotate", "medium", 0.8, _f_rotate),
    FamilySpec("digit_ops", "medium", 0.9, _f_digit_ops),
    # hard / mini-codebases
    FamilySpec("two_sum", "hard", 1.0, _f_two_sum),
    FamilySpec("binary_search", "hard", 0.9, _f_binary_search),
    FamilySpec("merge_sorted", "hard", 0.9, _f_merge_sorted),
    FamilySpec("mini_stats", "hard", 1.1, _f_mini_codebase_stats),
    FamilySpec("mini_text", "hard", 1.1, _f_mini_codebase_text),
    FamilySpec("flatten", "hard", 0.8, _f_matrix_flatten),
)

_DEFAULT_MIX = {"easy": 0.40, "medium": 0.40, "hard": 0.20}


def parse_mix(spec: str | dict[str, float] | None) -> dict[str, float]:
    if spec is None:
        return dict(_DEFAULT_MIX)
    if isinstance(spec, dict):
        raw = {str(k).lower(): float(v) for k, v in spec.items()}
    else:
        raw = {}
        for part in str(spec).split(","):
            part = part.strip()
            if not part:
                continue
            if ":" not in part:
                raise ValueError(f"bad mix fragment {part!r}; expected tier:weight")
            tier, weight = part.split(":", 1)
            raw[tier.strip().lower()] = float(weight)
    for tier in raw:
        if tier not in {"easy", "medium", "hard"}:
            raise ValueError(f"unknown complexity tier {tier!r}")
    total = sum(raw.values())
    if total <= 0:
        raise ValueError("mix weights must sum to > 0")
    return {k: v / total for k, v in raw.items()}


def _families_for_tier(tier: str) -> list[FamilySpec]:
    return [f for f in FAMILIES if f.complexity == tier]


def _pick_family(rng: random.Random, tier: str) -> FamilySpec:
    pool = _families_for_tier(tier)
    if not pool:
        raise ValueError(f"no families for tier {tier!r}")
    weights = [f.weight for f in pool]
    return rng.choices(pool, weights=weights, k=1)[0]


def generate_grounded_task(
    *,
    seed: int,
    index: int,
    tier: str,
    family_name: str | None = None,
) -> CodeTask:
    """Generate one unit-test-grounded task (solution first, tests from execution)."""
    rng = _stable_rng("corpus", str(seed), tier, str(index), family_name or "")
    if family_name:
        matches = [f for f in FAMILIES if f.name == family_name]
        if not matches:
            raise ValueError(f"unknown family {family_name!r}")
        fam = matches[0]
        tier = fam.complexity
    else:
        fam = _pick_family(rng, tier)
    prompt, solution, call_exprs, compare, tags = fam.generate(rng, index)
    # Ensure enough unique calls.
    unique_calls: list[str] = []
    seen: set[str] = set()
    for c in call_exprs:
        if c not in seen:
            unique_calls.append(c)
            seen.add(c)
    if len(unique_calls) < 3:
        # Pad with re-samples by re-running family with secondary rng.
        pad_rng = _stable_rng("corpus-pad", str(seed), str(index), fam.name)
        _, _, more, _, _ = fam.generate(pad_rng, index + 10_000)
        for c in more:
            if c not in seen:
                unique_calls.append(c)
                seen.add(c)
            if len(unique_calls) >= 4:
                break
    cases = ground_cases(solution, unique_calls, compare=compare)
    # Require diversity of call sites.
    if len({c for c, _ in cases}) < 2:
        raise ValueError(f"insufficient call diversity for {fam.name}#{index}")
    task_id = f"{fam.name}_{tier}_{seed}_{index}"
    tag_list = tuple(dict.fromkeys((*tags, tier, fam.name, "corpus")))
    task = CodeTask(
        task_id=task_id,
        prompt=prompt,
        solution=solution if solution.endswith("\n") else solution + "\n",
        cases=cases,
        compare=compare if compare in {"==", "is"} else "==",
        tags=tag_list,
        timeout_s=3.0 if tier != "hard" else 4.0,
    )
    return task


def generate_code_corpus(
    *,
    seed: int = 0,
    count: int = 500,
    mix: str | dict[str, float] | None = None,
    verify: bool = True,
    include_hand_catalog: bool = False,
) -> list[CodeTask]:
    """Generate ``count`` unit-test-grounded tasks across complexity mix.

    Parameters
    ----------
    seed:
        Deterministic corpus seed.
    count:
        Number of generated tasks (not including optional hand catalog).
    mix:
        Complexity mix, e.g. ``\"easy:0.4,medium:0.4,hard:0.2\"``.
    verify:
        Re-check each golden via sandboxed verifier (fail-closed skip).
    include_hand_catalog:
        Prepend the small hand-authored smoke catalog when True.
    """
    if count < 0:
        raise ValueError("count must be non-negative")
    ratios = parse_mix(mix)
    tiers = list(ratios.keys())
    weights = [ratios[t] for t in tiers]
    tasks: list[CodeTask] = []
    if include_hand_catalog:
        from seiso.rl_verify.synth_code import _base_catalog

        tasks.extend(_base_catalog())

    tier_rng = _stable_rng("corpus-tiers", str(seed), str(count))
    index = 0
    generated = 0
    attempts = 0
    max_attempts = max(count * 8, count + 100)
    while generated < count and attempts < max_attempts:
        attempts += 1
        tier = tier_rng.choices(tiers, weights=weights, k=1)[0]
        try:
            task = generate_grounded_task(seed=seed, index=index, tier=tier)
            index += 1
            if verify:
                validate_task(task)
            tasks.append(task)
            generated += 1
        except Exception:
            index += 1
            continue
    if generated < count:
        raise RuntimeError(
            f"only generated {generated}/{count} grounded tasks after "
            f"{attempts} attempts (seed={seed}); try a different seed or mix"
        )
    return tasks


def corpus_stats(tasks: Sequence[CodeTask]) -> dict[str, Any]:
    by_tier: dict[str, int] = {"easy": 0, "medium": 0, "hard": 0, "other": 0}
    for task in tasks:
        tier = "other"
        for t in ("easy", "medium", "hard"):
            if t in task.tags:
                tier = t
                break
        by_tier[tier] = by_tier.get(tier, 0) + 1
    return {
        "tasks": len(tasks),
        "by_complexity": by_tier,
        "families": sorted({t.tags[-2] if len(t.tags) >= 2 else "?" for t in tasks}),
    }
