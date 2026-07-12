#!/usr/bin/env python3
"""Split an elite/multi-domain coding JSONL root into specialist train/bench shards.

Expects under --data-root:
  train.jsonl, bench.jsonl  (from prepare_elite_coding_mix.py)

Writes under --data-root/specialists/:
  warmup.jsonl
  function_{easy,medium,all}.jsonl
  contest_{easy,medium,hard,all}.jsonl
  codebase_{easy,medium,hard,all}.jsonl
  bench_{function,contest,codebase,all}.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path


def _load(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write(path: Path, rows: list[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def _split_diff(rows: list[dict]) -> dict[str, list[dict]]:
    rows = sorted(rows, key=lambda r: float(r.get("difficulty") or 0))
    n = len(rows)
    if n == 0:
        return {"easy": [], "medium": [], "hard": [], "all": []}
    return {
        "easy": rows[: max(1, int(n * 0.4))],
        "medium": rows[int(n * 0.15) : max(int(n * 0.15) + 1, int(n * 0.75))],
        "hard": rows[int(n * 0.35) :],
        "all": rows,
    }


def _pad(bench_rows: list[dict], pool: list[dict], n: int, rng: random.Random) -> list[dict]:
    seen = {r.get("hash_id") for r in bench_rows}
    out = list(bench_rows)
    for r in reversed(pool):
        h = r.get("hash_id")
        if h in seen:
            continue
        out.append(r)
        seen.add(h)
        if len(out) >= n:
            break
    rng.shuffle(out)
    return out[:n]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/elite_coding"))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--bench-size", type=int, default=48)
    parser.add_argument("--bench-all-size", type=int, default=80)
    args = parser.parse_args()
    rng = random.Random(args.seed)

    root = args.data_root
    train = _load(root / "train.jsonl")
    bench = _load(root / "bench.jsonl") if (root / "bench.jsonl").exists() else []
    if not train:
        raise SystemExit(f"missing train.jsonl under {root}")

    by_dom: dict[str, list[dict]] = {
        "contest": [],
        "codebase": [],
        "function": [],
        "library": [],
    }
    for r in train:
        d = str(r.get("domain") or "contest")
        by_dom.setdefault(d, []).append(r)

    function_all = by_dom.get("function", []) + by_dom.get("library", [])
    contest = by_dom.get("contest", [])
    codebase = by_dom.get("codebase", [])

    fn_s, cb_s, ct_s = _split_diff(function_all), _split_diff(codebase), _split_diff(contest)
    warmup = fn_s["easy"][:1200] + cb_s["easy"][:800] + ct_s["easy"][:600]
    rng.shuffle(warmup)

    bench_by: dict[str, list[dict]] = {
        "contest": [],
        "codebase": [],
        "function": [],
        "library": [],
    }
    for r in bench:
        d = str(r.get("domain") or "contest")
        bench_by.setdefault(d, []).append(r)

    out = root / "specialists"
    stats: dict[str, int] = {}
    stats["warmup"] = _write(out / "warmup.jsonl", warmup)
    for name, rows in [
        ("function_easy", fn_s["easy"]),
        ("function_medium", fn_s["medium"]),
        ("function_all", fn_s["all"]),
        ("contest_easy", ct_s["easy"]),
        ("contest_medium", ct_s["medium"]),
        ("contest_hard", ct_s["hard"]),
        ("contest_all", ct_s["all"]),
        ("codebase_easy", cb_s["easy"]),
        ("codebase_medium", cb_s["medium"]),
        ("codebase_hard", cb_s["hard"]),
        ("codebase_all", cb_s["all"]),
    ]:
        stats[name] = _write(out / f"{name}.jsonl", rows)

    bf = _pad(
        (bench_by.get("function") or []) + (bench_by.get("library") or []),
        function_all,
        args.bench_size,
        rng,
    )
    bc = _pad(bench_by.get("contest") or [], contest, args.bench_size, rng)
    bb = _pad(bench_by.get("codebase") or [], codebase, args.bench_size, rng)
    ba = _pad(bench, train, args.bench_all_size, rng)
    stats["bench_function"] = _write(out / "bench_function.jsonl", bf)
    stats["bench_contest"] = _write(out / "bench_contest.jsonl", bc)
    stats["bench_codebase"] = _write(out / "bench_codebase.jsonl", bb)
    stats["bench_all"] = _write(out / "bench_all.jsonl", ba)

    meta = {
        "stats": stats,
        "recipe": "warmup -> function -> contest -> package",
        "domains": {k: len(v) for k, v in by_dom.items()},
        "reward_train": dict(Counter(r.get("reward_name") for r in train)),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
