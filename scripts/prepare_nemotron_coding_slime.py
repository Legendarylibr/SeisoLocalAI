#!/usr/bin/env python3
"""Convert nvidia/Nemotron-RL-coding-competitive_coding → slime JSONL.

Output rows:
  {
    "prompt": "<user content>",
    "unit_tests": {"inputs": [...], "outputs": [...]},
    "hash_id": "...",
    "source": "...",
    "dataset": "..."
  }
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _extract_prompt(row: dict) -> str | None:
    rcp = row.get("responses_create_params") or {}
    if not isinstance(rcp, dict):
        return None
    messages = rcp.get("input") or []
    if not isinstance(messages, list) or not messages:
        return None
    # Prefer last user message; fall back to first contentful message.
    for msg in reversed(messages):
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "user" and msg.get("content"):
            return str(msg["content"]).strip()
    for msg in messages:
        if isinstance(msg, dict) and msg.get("content"):
            return str(msg["content"]).strip()
    return None


def _extract_unit_tests(row: dict, max_tests: int) -> dict | None:
    vm = row.get("verifier_metadata") or {}
    if not isinstance(vm, dict):
        return None
    ut = vm.get("unit_tests") or {}
    if not isinstance(ut, dict):
        return None
    inputs = ut.get("inputs") or []
    outputs = ut.get("outputs") or []
    if not inputs or not outputs or len(inputs) != len(outputs):
        return None
    n = min(len(inputs), max_tests)
    return {
        "inputs": [str(x) for x in inputs[:n]],
        "outputs": [str(x) for x in outputs[:n]],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        default="nvidia/Nemotron-RL-coding-competitive_coding",
        help="HF dataset repo id",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/nemotron_competitive_coding_slime.jsonl"),
        help="Output JSONL path",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--max-tests-per-sample",
        type=int,
        default=8,
        help="Cap unit tests stored per row (speeds reward eval)",
    )
    parser.add_argument(
        "--streaming",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    from datasets import load_dataset

    print(f"Loading {args.repo} (streaming={args.streaming})...", file=sys.stderr)
    ds = load_dataset(args.repo, split="train", streaming=args.streaming)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    skipped = 0
    with args.output.open("w", encoding="utf-8") as out:
        for row in ds:
            prompt = _extract_prompt(row)
            unit_tests = _extract_unit_tests(row, args.max_tests_per_sample)
            if not prompt or not unit_tests:
                skipped += 1
                continue
            record = {
                "prompt": prompt,
                "answer": "",  # verifiable via unit_tests, not string match
                "unit_tests": unit_tests,
                "hash_id": row.get("hash_id"),
                "source": row.get("source"),
                "dataset": row.get("dataset"),
                "max_unit_tests": args.max_tests_per_sample,
                "unit_test_timeout_sec": 2.0,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1
            if kept % 500 == 0:
                print(f"  wrote {kept} rows...", file=sys.stderr)
            if args.max_samples is not None and kept >= args.max_samples:
                break

    print(
        f"Done: kept={kept} skipped={skipped} → {args.output}",
        file=sys.stderr,
    )
    return 0 if kept else 1


if __name__ == "__main__":
    raise SystemExit(main())
