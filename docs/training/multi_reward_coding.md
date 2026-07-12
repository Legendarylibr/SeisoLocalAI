# Multi-reward coding RL (slime)

Seiso slime can train on **several verifiable coding domains in one stack**:

| `reward` / sample `reward_name` | Domain | Judge |
|---|---|---|
| `unit_tests` | Competitive I/O | stdin/stdout unit tests |
| `codebase_tests` | Multi-file packages | pytest project harness |
| `assert_tests` | Functions / libraries | assert list or `check(fn)` |
| `multi` / `auto` / `mixed` | Mixed JSONL | per-sample dispatch |

Use `reward: multi` in the train YAML. Each JSONL row should set `reward_name` (or include the fields the dispatcher detects: `unit_tests`, `codebase`, `assert_tests` / `entry_point`+`test`).

## Portable base config

See [configs/examples/slime_coding_multi_reward.yaml](../../configs/examples/slime_coding_multi_reward.yaml).

## Specialist recipe (beat-base loop)

Warm-up → **function** specialist → **contest** specialist → **package** specialist:

```bash
# 1) Build multi-domain data (HF + synthetic packages)
python scripts/prepare_elite_coding_mix.py --out-dir data/elite_coding

# 2) Specialist shards (warmup / domain trains / domain benches)
python scripts/build_coding_specialist_shards.py --data-root data/elite_coding

# 3) Run recipe-driven loop (does not require absolute machine paths)
python scripts/slime_coding_train_until_good.py \
  --mode specialist \
  --base-config configs/examples/slime_coding_multi_reward.yaml \
  --recipe configs/examples/slime_coding_specialist_recipe.yaml \
  --data-root data/elite_coding \
  --work-dir outputs/slime-specialist \
  --bench-limit 48 \
  --max-rounds 7
```

Recipe format: [configs/examples/slime_coding_specialist_recipe.yaml](../../configs/examples/slime_coding_specialist_recipe.yaml).

## Data prep scripts

| Script | Role |
|---|---|
| `scripts/generate_synthetic_codebases.py` | Thousands of multi-file pytest packages |
| `scripts/prepare_elite_coding_mix.py` | Contest + packages + MBPP/HumanEval/BCB mix |
| `scripts/build_coding_specialist_shards.py` | Domain curriculum + benches under `specialists/` |
| `scripts/benchmark_slime_coding.py` | Multi-reward held-out bench |
| `scripts/benchmark_slime_codebases.py` | Package-only / gold QA |

## Notes

- Prefer **held-out** checkpoint selection (loop keeps global + phase best).
- Specialist mode selects by domain `select_reward` when set.
- Large JSONL corpora under `data/` are **not** committed; regenerate with the scripts above.
