#!/usr/bin/env python3
"""Train Seiso slime on competitive coding, benchmark, retrain until targets met.

Targets (override via flags):
  - mean unit-test pass rate on held-out bench
  - code extraction rate (model emits ```python)
  - relative improvement over base baseline

Specialist loop (default, recommended to beat base):
  warm-up → function (assert_tests) → contest (unit_tests) → package (codebase_tests)

Phases can be fully described by a portable recipe YAML:
  configs/examples/slime_coding_specialist_recipe.yaml

See docs/training/multi_reward_coding.md.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

_SOURCE_RANK = {
    "aizu": 0,
    "atcoder": 1,
    "hackerearth": 2,
    "codechef": 3,
    "codeforces": 4,
}


def _run(cmd: list[str], *, env: dict | None = None, cwd: Path | None = None) -> int:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=cwd or ROOT, env=env)
    return int(proc.returncode)


def _load_report(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_round_config(
    base_cfg: dict,
    *,
    round_idx: int,
    out_dir: Path,
    overrides: dict,
) -> Path:
    cfg = deepcopy(base_cfg)
    cfg.update(overrides)
    cfg["output_dir"] = str(out_dir / f"round{round_idx}")
    path = out_dir / f"config_round{round_idx}.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def _is_good(report: dict, args: argparse.Namespace, baseline: dict | None) -> tuple[bool, str]:
    mean = float(report["mean_pass_rate"])
    code = float(report["code_extract_rate"])
    any_pass = float(report["any_pass_rate"])

    reasons_ok: list[str] = []
    reasons_fail: list[str] = []

    if mean >= args.target_mean_pass:
        reasons_ok.append(f"mean_pass={mean:.3f}>={args.target_mean_pass}")
    else:
        reasons_fail.append(f"mean_pass={mean:.3f}<{args.target_mean_pass}")

    if code >= args.target_code_rate:
        reasons_ok.append(f"code_rate={code:.3f}>={args.target_code_rate}")
    else:
        reasons_fail.append(f"code_rate={code:.3f}<{args.target_code_rate}")

    if any_pass >= args.target_any_pass:
        reasons_ok.append(f"any_pass={any_pass:.3f}>={args.target_any_pass}")
    else:
        reasons_fail.append(f"any_pass={any_pass:.3f}<{args.target_any_pass}")

    if baseline is not None and args.require_vs_baseline:
        base_mean = float(baseline["mean_pass_rate"])
        need = base_mean + args.min_abs_gain
        if args.min_rel_gain > 1.0 and base_mean > 1e-6:
            need = max(need, base_mean * args.min_rel_gain)
        if mean >= need:
            reasons_ok.append(f"vs_baseline mean {mean:.3f}>={need:.3f}")
        else:
            reasons_fail.append(
                f"vs_baseline mean {mean:.3f}<{need:.3f} (base={base_mean:.3f})"
            )

    ok = not reasons_fail
    detail = "; ".join(reasons_ok + reasons_fail)
    return ok, detail


def _has_adapter(path: Path) -> bool:
    return (
        (path / "adapter_model.safetensors").exists()
        or (path / "adapter_config.json").exists()
        or (path / "pytorch_model.bin").exists()
        or any(path.glob("*.safetensors"))
    )


def _difficulty_score(row: dict) -> float:
    """Lower = easier. Prefer precomputed mix difficulty when present."""
    if row.get("difficulty") is not None:
        try:
            return float(row["difficulty"])
        except (TypeError, ValueError):
            pass
    source = str(row.get("source") or "").lower()
    src = float(_SOURCE_RANK.get(source, 3))
    # Math without unit tests tends to give denser early signal → treat as easier.
    if str(row.get("domain") or row.get("reward_name") or "").lower() in {
        "math",
        "gsm8k",
        "mathematics",
    }:
        src = min(src, 1.0)
    ut = row.get("unit_tests") or {}
    n_tests = float(len(ut.get("inputs") or []))
    prompt_len = float(len(row.get("prompt") or ""))
    return src * 1_000_000.0 + n_tests * 1_000.0 + prompt_len


def _build_curriculum_shards(
    train_dataset: Path,
    work: Path,
    *,
    seed: int = 17,
) -> dict[str, Path]:
    """Write easy/medium/hard JSONL shards under work/curriculum/."""
    out_dir = work / "curriculum"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    with train_dataset.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"empty train dataset: {train_dataset}")

    ranked = sorted(rows, key=_difficulty_score)
    n = len(ranked)
    # Overlapping bands so later rounds still see some easy wins.
    bands = {
        "easy": ranked[: max(1, int(n * 0.40))],
        "medium": ranked[int(n * 0.15) : max(int(n * 0.15) + 1, int(n * 0.70))],
        "hard": ranked[int(n * 0.35) :],
        "mixed": ranked,  # full set for final polish
    }

    paths: dict[str, Path] = {}
    rng = random.Random(seed)
    for name, band in bands.items():
        path = out_dir / f"train_{name}.jsonl"
        shuffled = list(band)
        rng.shuffle(shuffled)
        with path.open("w", encoding="utf-8") as f:
            for row in shuffled:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        paths[name] = path
        print(
            f"curriculum {name}: n={len(shuffled)} -> {path}",
            flush=True,
        )
    meta = {
        "n_total": n,
        "n_easy": len(bands["easy"]),
        "n_medium": len(bands["medium"]),
        "n_hard": len(bands["hard"]),
        "sources_easy_top": _source_counts(bands["easy"]),
        "sources_hard_top": _source_counts(bands["hard"]),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return paths


def _source_counts(rows: list[dict], top: int = 5) -> list[list]:
    from collections import Counter

    c = Counter(str(r.get("source") or "?") for r in rows)
    return [[k, v] for k, v in c.most_common(top)]


def _pick_resume_adapter(round_out: Path, reports: list[tuple[Path, dict]]) -> Path | None:
    """Prefer the held-out best candidate; fall back to checkpoint-best / final."""
    if reports:
        best_path, _ = max(reports, key=lambda item: item[1]["mean_pass_rate"])
        if _has_adapter(best_path):
            return best_path
    for candidate in (round_out / "checkpoint-best", round_out):
        if _has_adapter(candidate):
            return candidate
    numbered = sorted(
        round_out.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1,
    )
    for candidate in reversed(numbered):
        if candidate.name == "checkpoint-best":
            continue
        if _has_adapter(candidate):
            return candidate
    return None


def _collect_candidates(round_out: Path, *, max_numbered: int = 5) -> list[Path]:
    """Final + EMA best + evenly spaced numbered checkpoints (cap cost)."""
    candidates: list[Path] = []
    if _has_adapter(round_out):
        candidates.append(round_out)
    best_dir = round_out / "checkpoint-best"
    if best_dir.exists() and best_dir.resolve() != round_out.resolve() and _has_adapter(best_dir):
        candidates.append(best_dir)

    numbered = sorted(
        (
            p
            for p in round_out.glob("checkpoint-*")
            if p.name != "checkpoint-best" and p.name.split("-")[-1].isdigit()
        ),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    if numbered:
        if len(numbered) <= max_numbered:
            pick = numbered
        else:
            # Always include earliest, mid, latest + evenly spaced.
            idxs = sorted(
                {
                    0,
                    len(numbered) // 4,
                    len(numbered) // 2,
                    (3 * len(numbered)) // 4,
                    len(numbered) - 1,
                }
            )
            pick = [numbered[i] for i in idxs]
        for p in pick:
            if p not in candidates and _has_adapter(p):
                candidates.append(p)
    return candidates


def _coding_metric(report: dict, select_reward: str | None = None) -> float:
    """Held-out selection metric. Optionally filter by reward_name in results."""
    results = report.get("results") or []
    if select_reward and results:
        scores = [
            float(r.get("pass_rate") or 0.0)
            for r in results
            if str(r.get("reward_name") or "") == select_reward
            or (
                select_reward == "assert_tests"
                and str(r.get("reward_name") or "") in {"assert_tests"}
            )
            or (
                select_reward == "unit_tests"
                and str(r.get("reward_name") or "") == "unit_tests"
            )
            or (
                select_reward == "codebase_tests"
                and str(r.get("reward_name") or "") == "codebase_tests"
            )
        ]
        if scores:
            return sum(scores) / len(scores)
    if report.get("coding_mean") is not None:
        return float(report["coding_mean"])
    return float(report.get("mean_pass_rate") or 0.0)


def _resolve_data_path(path: str | Path, data_root: Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    # Prefer data_root for specialist-relative paths
    cand = data_root / p
    if cand.exists() or not (ROOT / p).exists():
        return cand
    return ROOT / p


def _schedules_from_recipe(recipe_path: Path, data_root: Path) -> list[dict]:
    """Load portable specialist phases from YAML recipe."""
    raw = yaml.safe_load(recipe_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict) or not raw.get("phases"):
        raise ValueError(f"invalid recipe (need phases[]): {recipe_path}")
    defaults = dict(raw.get("defaults") or {})
    schedules: list[dict] = []
    for phase in raw["phases"]:
        if not isinstance(phase, dict):
            continue
        item = {**defaults, **phase}
        if item.get("dataset"):
            item["dataset"] = str(_resolve_data_path(item["dataset"], data_root))
        if item.get("bench_dataset"):
            item["bench_dataset"] = str(
                _resolve_data_path(item["bench_dataset"], data_root)
            )
        # normalize null select_reward
        if item.get("select_reward") in {"", "null", "None"}:
            item["select_reward"] = None
        schedules.append(item)
    if not schedules:
        raise ValueError(f"recipe has no phases: {recipe_path}")
    print(
        f"loaded recipe {recipe_path.name}: {len(schedules)} phases "
        f"({raw.get('name') or 'unnamed'})",
        flush=True,
    )
    return schedules


def _optimal_schedules(
    *,
    mode: str = "coding",
    data_root: Path | None = None,
    recipe: Path | None = None,
) -> list[dict]:
    """Round schedules.

    mode=specialist (recommended to beat base):
      warm-up → function → contest → package (from recipe YAML if provided)
    mode=coding: multi-domain mix rounds
    """
    common = {
        "rollouts_per_prompt": 4,
        "rollout_batch_size": 4,
        "policy_micro_batch_size": 1,
        "train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "max_grad_norm": 1.0,
        "epochs": 1,
        "top_p": 0.95,
        "max_vram_gb": 22.5,
        "auto_stop": True,
        "auto_stop_metric": "reward_mean",
        "auto_stop_ema_alpha": 0.25,
        "save_steps": 5,
        "logging_steps": 1,
        "process_reward_weight": 0.0,
        "missing_thinking_penalty": 0.0,
        "require_thinking_trace": False,
        "sequential_rollouts": True,
        "stop_on_nonfinite": True,
    }
    root = data_root or (ROOT / "data/elite_coding")
    default_recipe = ROOT / "configs/examples/slime_coding_specialist_recipe.yaml"

    if mode == "specialist":
        recipe_path = recipe or default_recipe
        if recipe_path is not None and recipe_path.exists():
            return _schedules_from_recipe(recipe_path, root)
        raise FileNotFoundError(
            f"specialist mode needs a recipe YAML (tried {recipe_path}). "
            "Pass --recipe configs/examples/slime_coding_specialist_recipe.yaml"
        )

    if mode == "coding":
        return [
            {
                **common,
                "phase": "warmup",
                "curriculum": "warmup",
                "reward": "multi",
                "max_samples_per_epoch": 200,
                "max_prompt_tokens": 1280,
                "max_new_tokens": 768,
                "learning_rate": 6.0e-6,
                "kl_coef": 0.0,
                "temperature": 1.1,
                "auto_stop_patience": 10,
                "auto_stop_min_delta": 0.015,
                "auto_stop_warmup_steps": 5,
                "thinking_instruction": (
                    "Elite coding: functions in ```python; packages with path=; "
                    "contest stdin/stdout in ```python."
                ),
                "extra": {"max_steps": 18, "sequential_rollouts": True},
            },
            {
                **common,
                "phase": "mixed",
                "curriculum": "easy",
                "reward": "multi",
                "max_samples_per_epoch": 280,
                "max_prompt_tokens": 1536,
                "max_new_tokens": 1024,
                "learning_rate": 5.0e-6,
                "kl_coef": 0.0,
                "temperature": 1.05,
                "auto_stop_patience": 14,
                "auto_stop_min_delta": 0.01,
                "auto_stop_warmup_steps": 8,
                "thinking_instruction": (
                    "For contest: ```python stdin/stdout. "
                    "For packages: ```python path=.... "
                    "For functions: full ```python defs."
                ),
                "extra": {"max_steps": 40, "sequential_rollouts": True},
            },
            {
                **common,
                "phase": "mixed",
                "curriculum": "medium",
                "reward": "multi",
                "max_samples_per_epoch": 320,
                "max_prompt_tokens": 1536,
                "max_new_tokens": 1152,
                "learning_rate": 3.0e-6,
                "kl_coef": 0.01,
                "temperature": 0.95,
                "auto_stop_patience": 12,
                "auto_stop_min_delta": 0.008,
                "auto_stop_warmup_steps": 6,
                "thinking_instruction": (
                    "For contest: ```python stdin/stdout. "
                    "For packages: ```python path=.... "
                    "For functions: full ```python defs."
                ),
                "extra": {"max_steps": 45, "sequential_rollouts": True},
            },
            {
                **common,
                "phase": "mixed",
                "curriculum": "hard",
                "reward": "multi",
                "max_samples_per_epoch": 360,
                "max_prompt_tokens": 1792,
                "max_new_tokens": 1280,
                "learning_rate": 2.0e-6,
                "kl_coef": 0.02,
                "temperature": 0.9,
                "auto_stop_patience": 12,
                "auto_stop_min_delta": 0.006,
                "auto_stop_warmup_steps": 6,
                "thinking_instruction": (
                    "For contest: ```python stdin/stdout. "
                    "For packages: ```python path=.... "
                    "For functions: full ```python defs."
                ),
                "extra": {"max_steps": 50, "sequential_rollouts": True},
            },
        ]
    # Legacy multi-balanced schedules (not coding-specialized).
    return [
        {
            **common,
            "curriculum": "easy",
            "max_samples_per_epoch": 280,
            "max_prompt_tokens": 896,
            "max_new_tokens": 576,
            "learning_rate": 6.0e-6,
            "kl_coef": 0.0,
            "temperature": 1.1,
            "auto_stop_patience": 16,
            "auto_stop_min_delta": 0.012,
            "auto_stop_warmup_steps": 8,
            "extra": {"max_steps": 40, "sequential_rollouts": True},
        },
        {
            **common,
            "curriculum": "medium",
            "max_samples_per_epoch": 360,
            "max_prompt_tokens": 960,
            "max_new_tokens": 640,
            "learning_rate": 3.5e-6,
            "kl_coef": 0.01,
            "temperature": 1.0,
            "auto_stop_patience": 14,
            "auto_stop_min_delta": 0.01,
            "auto_stop_warmup_steps": 8,
            "extra": {"max_steps": 45, "sequential_rollouts": True},
        },
        {
            **common,
            "curriculum": "hard",
            "max_samples_per_epoch": 400,
            "max_prompt_tokens": 1024,
            "max_new_tokens": 704,
            "learning_rate": 2.0e-6,
            "kl_coef": 0.02,
            "temperature": 0.9,
            "auto_stop_patience": 12,
            "auto_stop_min_delta": 0.008,
            "auto_stop_warmup_steps": 6,
            "extra": {"max_steps": 50, "sequential_rollouts": True},
        },
        {
            **common,
            "curriculum": "mixed",
            "max_samples_per_epoch": 420,
            "max_prompt_tokens": 1024,
            "max_new_tokens": 704,
            "learning_rate": 1.5e-6,
            "kl_coef": 0.02,
            "temperature": 0.85,
            "auto_stop_patience": 12,
            "auto_stop_min_delta": 0.006,
            "auto_stop_warmup_steps": 6,
            "extra": {"max_steps": 50, "sequential_rollouts": True},
        },
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-config",
        type=Path,
        default=ROOT / "configs/examples/slime_coding_multi_reward.yaml",
    )
    parser.add_argument(
        "--recipe",
        type=Path,
        default=ROOT / "configs/examples/slime_coding_specialist_recipe.yaml",
        help="Specialist phase recipe YAML (portable paths under --data-root)",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=ROOT / "outputs/slime-coding-specialist",
    )
    parser.add_argument(
        "--bench-dataset",
        type=Path,
        default=ROOT / "data/elite_coding/bench.jsonl",
    )
    parser.add_argument(
        "--train-dataset",
        type=Path,
        default=ROOT / "data/elite_coding/train.jsonl",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT / "data/elite_coding",
        help="Prebuilt curriculum dir (train_easy/medium/hard/mixed + warmup_multi_easy)",
    )
    parser.add_argument(
        "--mode",
        choices=("specialist", "coding", "balanced"),
        default="specialist",
        help=(
            "specialist = warm-up → function → contest → package (beat base); "
            "coding = multi-domain mix; balanced = generic mix"
        ),
    )
    parser.add_argument("--bench-limit", type=int, default=48)
    parser.add_argument("--max-rounds", type=int, default=7)
    parser.add_argument("--target-mean-pass", type=float, default=0.36)
    parser.add_argument("--target-code-rate", type=float, default=0.90)
    parser.add_argument("--target-any-pass", type=float, default=0.60)
    parser.add_argument("--require-vs-baseline", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-abs-gain", type=float, default=0.015)
    parser.add_argument("--min-rel-gain", type=float, default=1.04)
    parser.add_argument(
        "--stale-rounds",
        type=int,
        default=2,
        help="Stop early if held-out best does not improve for this many consecutive rounds",
    )
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-curriculum", action="store_true")
    parser.add_argument("--skip-warmup", action="store_true", help="Skip multi warm-up round")
    parser.add_argument("--python", default=str(ROOT / ".venv/bin/python"))
    parser.add_argument("--seiso", default=str(ROOT / ".venv/bin/seiso"))
    args = parser.parse_args()

    work = args.work_dir
    work.mkdir(parents=True, exist_ok=True)

    base_cfg = yaml.safe_load(args.base_config.read_text(encoding="utf-8"))
    # Default train path; per-round curriculum overrides this.
    base_cfg["dataset"] = str(args.train_dataset)
    # Keep config reward when already multi/codebase-capable; only force unit_tests
    # for legacy pure-contest coding configs.
    if args.mode == "coding" and str(base_cfg.get("reward") or "") not in {
        "multi",
        "auto",
        "mixed",
        "codebase_tests",
    }:
        base_cfg["reward"] = "unit_tests"

    env = os.environ.copy()
    env.setdefault("SEISO_NVIDIA_HOST_VENV_ACK", "1")
    env.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    env.setdefault("OMP_NUM_THREADS", "4")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    history: list[dict] = []
    baseline: dict | None = None
    base_report_path = work / "baseline_bench.json"

    curriculum_paths: dict[str, Path] | None = None
    data_root = Path(args.data_root)
    prebuilt = {
        "easy": data_root / "train_easy.jsonl",
        "medium": data_root / "train_medium.jsonl",
        "hard": data_root / "train_hard.jsonl",
        "mixed": data_root / "train_mixed.jsonl",
        "warmup": data_root / "warmup_multi_easy.jsonl",
    }
    if not args.skip_curriculum and all(
        prebuilt[k].exists() for k in ("easy", "medium", "hard", "mixed")
    ):
        curriculum_paths = {k: v for k, v in prebuilt.items() if v.exists()}
        print(
            "Using prebuilt coding curriculum: "
            + ", ".join(f"{k}={v}" for k, v in curriculum_paths.items()),
            flush=True,
        )
    elif not args.skip_curriculum:
        print("Building curriculum shards...", flush=True)
        curriculum_paths = _build_curriculum_shards(
            Path(args.train_dataset),
            work,
            seed=int(base_cfg.get("seed") or 17),
        )
        # Optional warm-up file if present next to train dataset.
        wu = data_root / "warmup_multi_easy.jsonl"
        if wu.exists():
            curriculum_paths["warmup"] = wu

    if args.skip_baseline and base_report_path.exists():
        baseline = _load_report(base_report_path)
        print(
            f"BASELINE (cached) mean_pass={baseline['mean_pass_rate']:.3f} "
            f"code={baseline['code_extract_rate']:.3f} "
            f"any_pass={baseline['any_pass_rate']:.3f}",
            flush=True,
        )
        history.append(
            {
                "round": 0,
                "phase": "baseline_cached",
                **{
                    k: baseline[k]
                    for k in (
                        "mean_pass_rate",
                        "perfect_rate",
                        "any_pass_rate",
                        "code_extract_rate",
                    )
                },
            }
        )

    if baseline is None and not args.skip_baseline:
        baseline_ds = Path(args.bench_dataset)
        if args.mode == "specialist":
            cand = Path(args.data_root) / "specialists" / "bench_all.jsonl"
            if cand.exists():
                baseline_ds = cand
        rc = _run(
            [
                args.python,
                str(ROOT / "scripts/benchmark_slime_coding.py"),
                "--model-id",
                str(base_cfg["model_id"]),
                "--dataset",
                str(baseline_ds),
                "--limit",
                str(args.bench_limit),
                "--max-prompt-tokens",
                "1536",
                "--max-new-tokens",
                "1024",
                "--temperature",
                "0.2",
                "--dtype",
                str(base_cfg.get("dtype", "bfloat16")),
                "--no-require-thinking",
                "--output",
                str(base_report_path),
            ],
            env=env,
        )
        if rc != 0:
            print("baseline benchmark failed", file=sys.stderr)
            return rc
        baseline = _load_report(base_report_path)
        print(
            f"BASELINE mean_pass={baseline['mean_pass_rate']:.3f} "
            f"code={baseline['code_extract_rate']:.3f} "
            f"any_pass={baseline['any_pass_rate']:.3f}",
            flush=True,
        )
        history.append(
            {
                "round": 0,
                "phase": "baseline",
                **{
                    k: baseline[k]
                    for k in (
                        "mean_pass_rate",
                        "perfect_rate",
                        "any_pass_rate",
                        "code_extract_rate",
                    )
                },
            }
        )

    schedules = _optimal_schedules(
        mode=args.mode,
        data_root=Path(args.data_root),
        recipe=Path(args.recipe) if args.recipe else None,
    )
    # Cap schedules to max_rounds
    if len(schedules) > args.max_rounds:
        schedules = schedules[: args.max_rounds]
    if args.skip_warmup and schedules and schedules[0].get("phase") == "warmup":
        schedules = schedules[1:]
        print("skip-warmup: starting on first specialist phase", flush=True)

    best_report: dict | None = None
    best_ckpt: Path | None = None
    resume_adapter: Path | None = None
    stale = 0
    # Per-phase best (function/contest/codebase) — resume uses global best overall metric
    # but we also track phase bests for the summary.
    phase_best: dict[str, dict] = {}

    for round_idx in range(1, min(args.max_rounds, len(schedules)) + 1):
        schedule = deepcopy(schedules[round_idx - 1])
        curriculum_name = schedule.pop("curriculum", None)
        phase = str(schedule.pop("phase", curriculum_name or "coding"))
        select_reward = schedule.pop("select_reward", None)
        bench_override = schedule.pop("bench_dataset", None)

        # Explicit dataset in specialist mode; else curriculum map / default train.
        if schedule.get("dataset"):
            print(
                f"using phase={phase} dataset={schedule['dataset']} "
                f"reward={schedule.get('reward')} select={select_reward}",
                flush=True,
            )
        elif curriculum_paths and curriculum_name and curriculum_name in curriculum_paths:
            schedule["dataset"] = str(curriculum_paths[curriculum_name])
            print(f"using curriculum={curriculum_name} dataset={schedule['dataset']}", flush=True)
        elif curriculum_name == "warmup":
            wu = Path(args.data_root) / "specialists" / "warmup.jsonl"
            if not wu.exists():
                wu = Path(args.data_root) / "warmup_multi_easy.jsonl"
            schedule["dataset"] = str(wu if wu.exists() else args.train_dataset)
            schedule.setdefault("reward", "multi")
            print(f"warmup dataset={schedule['dataset']}", flush=True)
        else:
            schedule["dataset"] = str(args.train_dataset)

        if resume_adapter is not None:
            schedule["resume_from"] = str(resume_adapter)
            print(f"continuing from adapter: {resume_adapter}", flush=True)

        print(
            f"\n======== ROUND {round_idx}/{len(schedules)} phase={phase} ========",
            flush=True,
        )
        print(f"schedule: {json.dumps(schedule, default=str)}", flush=True)
        bench_path_this = Path(bench_override) if bench_override else Path(args.bench_dataset)

        cfg_path = _write_round_config(
            base_cfg,
            round_idx=round_idx,
            out_dir=work,
            overrides=schedule,
        )
        train_rc = _run([args.seiso, "train", "--config", str(cfg_path)], env=env)
        if train_rc != 0:
            print(f"training failed round {round_idx} rc={train_rc}", file=sys.stderr)
            history.append({"round": round_idx, "phase": "train_failed", "rc": train_rc})
            salvage = _pick_resume_adapter(work / f"round{round_idx}", [])
            if salvage is not None and best_ckpt is None:
                resume_adapter = salvage
            elif best_ckpt is not None:
                resume_adapter = best_ckpt
            continue

        round_out = work / f"round{round_idx}"
        candidates = _collect_candidates(round_out, max_numbered=5)
        print(
            f"evaluating {len(candidates)} candidates: "
            + ", ".join(str(c.relative_to(work) if c.is_relative_to(work) else c) for c in candidates),
            flush=True,
        )

        reports: list[tuple[Path, dict]] = []
        # Specialist phases use domain benches; default cap to full file if smaller.
        try:
            with bench_path_this.open(encoding="utf-8") as bf:
                n_bench_rows = sum(1 for line in bf if line.strip())
        except OSError:
            n_bench_rows = args.bench_limit
        bench_limit = min(args.bench_limit, n_bench_rows) if n_bench_rows else args.bench_limit

        for cand in candidates:
            tag = "final" if cand == round_out else cand.name
            bench_path = work / f"bench_round{round_idx}_{tag}.json"
            bench_rc = _run(
                [
                    args.python,
                    str(ROOT / "scripts/benchmark_slime_coding.py"),
                    "--model-id",
                    str(base_cfg["model_id"]),
                    "--adapter",
                    str(cand),
                    "--dataset",
                    str(bench_path_this),
                    "--limit",
                    str(bench_limit),
                    "--max-prompt-tokens",
                    "1536",
                    "--max-new-tokens",
                    "1024",
                    "--temperature",
                    "0.2",
                    "--dtype",
                    str(base_cfg.get("dtype", "bfloat16")),
                    "--no-require-thinking",
                    "--output",
                    str(bench_path),
                ],
                env=env,
            )
            if bench_rc != 0:
                print(f"benchmark failed round {round_idx} cand={cand}", file=sys.stderr)
                history.append(
                    {
                        "round": round_idx,
                        "phase": "bench_failed",
                        "adapter": str(cand),
                        "rc": bench_rc,
                    }
                )
                continue
            reports.append((cand, _load_report(bench_path)))

        if not reports:
            print(f"no usable checkpoints for round {round_idx}", file=sys.stderr)
            continue

        # Select by phase metric (domain-filtered when select_reward set).
        adapter, report = max(
            reports, key=lambda item: _coding_metric(item[1], select_reward)
        )
        metric_now = _coding_metric(report, select_reward)
        entry = {
            "round": round_idx,
            "phase": phase,
            "adapter": str(adapter),
            "mean_pass_rate": report["mean_pass_rate"],
            "coding_metric": metric_now,
            "select_reward": select_reward,
            "coding_mean": report.get("coding_mean"),
            "perfect_rate": report["perfect_rate"],
            "any_pass_rate": report["any_pass_rate"],
            "code_extract_rate": report["code_extract_rate"],
            "curriculum": curriculum_name,
            "bench_dataset": str(bench_path_this),
            "candidates": [
                {
                    "adapter": str(p),
                    "mean_pass_rate": r["mean_pass_rate"],
                    "coding_metric": _coding_metric(r, select_reward),
                    "perfect_rate": r["perfect_rate"],
                }
                for p, r in reports
            ],
        }
        history.append(entry)
        print(
            f"ROUND {round_idx} phase={phase} metric={metric_now:.3f} "
            f"mean_pass={report['mean_pass_rate']:.3f} "
            f"perfect={report['perfect_rate']:.3f} any_pass={report['any_pass_rate']:.3f} "
            f"code={report['code_extract_rate']:.3f} adapter={adapter}",
            flush=True,
        )

        # Track per-phase best
        prev_phase = phase_best.get(phase)
        if prev_phase is None or metric_now > float(prev_phase.get("metric", -1)) + 1e-6:
            phase_best[phase] = {
                "metric": metric_now,
                "adapter": str(adapter),
                "round": round_idx,
                "report": report,
            }
            print(f"new phase best phase={phase} metric={metric_now:.3f}", flush=True)

        improved = False
        best_metric = _coding_metric(best_report, None) if best_report else -1.0
        # Global best uses overall mean_pass on that phase's report (honest average).
        overall_now = float(report.get("mean_pass_rate") or 0.0)
        if best_report is None or overall_now > float(best_report.get("mean_pass_rate") or 0) + 1e-6:
            best_report = report
            best_ckpt = adapter
            improved = True
            stale = 0
            print(
                f"new global best mean_pass={overall_now:.3f} "
                f"(phase_metric={metric_now:.3f}) @ {best_ckpt}",
                flush=True,
            )
        else:
            if phase == "warmup":
                print(
                    "warmup did not beat global best (ok); continuing from global best",
                    flush=True,
                )
            else:
                # Stale counts only within non-warmup if phase metric didn't improve phase best
                # already recorded above; count stale when overall didn't improve.
                stale += 1
                print(
                    f"no global improvement (stale={stale}/{args.stale_rounds}); "
                    f"keeping {best_ckpt}",
                    flush=True,
                )

        # Resume from global best; if phase improved, prefer phase best adapter.
        phase_adapter = phase_best.get(phase, {}).get("adapter")
        if phase_adapter and _has_adapter(Path(phase_adapter)):
            resume_adapter = Path(phase_adapter)
        elif best_ckpt is not None and _has_adapter(best_ckpt):
            resume_adapter = best_ckpt
        else:
            resume_adapter = _pick_resume_adapter(round_out, reports)

        # Success judged on global best vs baseline (overall mean).
        if best_report is not None:
            ok, detail = _is_good(best_report, args, baseline)
        else:
            ok, detail = _is_good(report, args, baseline)
        print(f"quality check: ok={ok} | {detail}", flush=True)
        if ok and best_ckpt is not None:
            summary = {
                "status": "good",
                "winning_round": round_idx,
                "adapter": str(best_ckpt),
                "report": best_report,
                "baseline": baseline,
                "phase_best": {
                    k: {kk: vv for kk, vv in v.items() if kk != "report"}
                    for k, v in phase_best.items()
                },
                "history": history,
            }
            (work / "loop_summary.json").write_text(
                json.dumps(summary, indent=2), encoding="utf-8"
            )
            print(f"\nSUCCESS at round {round_idx}. Checkpoint: {best_ckpt}", flush=True)
            return 0

        # Specialist: don't early-stop the whole pipeline on one phase stall;
        # only stop if last 2 non-warmup rounds both failed global improvement.
        if args.mode != "specialist" and stale >= args.stale_rounds:
            print(
                f"\nEarly stop: no held-out improvement for {stale} rounds. "
                f"Best={best_ckpt}",
                flush=True,
            )
            break
        if args.mode == "specialist" and stale >= args.stale_rounds + 2:
            print(
                f"\nEarly stop (specialist): prolonged no global gain. Best={best_ckpt}",
                flush=True,
            )
            break

    summary = {
        "status": "not_met",
        "best_adapter": str(best_ckpt) if best_ckpt else None,
        "best_report": best_report,
        "baseline": baseline,
        "phase_best": {
            k: {kk: vv for kk, vv in v.items() if kk != "report"}
            for k, v in phase_best.items()
        },
        "history": history,
        "targets": {
            "mean_pass": args.target_mean_pass,
            "code_rate": args.target_code_rate,
            "any_pass": args.target_any_pass,
        },
    }
    (work / "loop_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nTargets not fully met; best checkpoint kept. See loop_summary.json", flush=True)
    if (
        baseline is not None
        and best_report is not None
        and _coding_metric(best_report) > _coding_metric(baseline) + 1e-6
    ):
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
