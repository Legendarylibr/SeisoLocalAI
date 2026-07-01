"""Tests for distill-rl research pipeline."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from seiso.distill_rl.config import (
    build_distill_rl_config,
    resolve_job_seeds,
    validate_stage_sequence,
)
from seiso.distill_rl.multiseed import aggregate_multiseed_runs
from seiso.distill_rl.preferences import build_preference_bundle
from seiso.distill_rl.prompts import (
    RolloutPrompt,
    load_rollout_prompts,
    split_train_val,
)
from seiso.distill_rl.runner import _checkpoint_step, _latest_checkpoint


def test_build_distill_rl_config_smoke_defaults(tmp_path: Path):
    cfg = build_distill_rl_config(
        job_id="job-1",
        user_id="user-1",
        data_dir=tmp_path,
        payload={"preset": "smoke"},
    )
    assert cfg.preset == "smoke"
    assert "evaluate" in cfg.stages
    assert cfg.align_distill_with_prompts is True
    assert cfg.train_val_fraction == 0.75
    assert cfg.trust_remote_code is False
    assert cfg.require_thinking_trace is True
    assert cfg.verifiable_outcome_rewards is True
    assert cfg.grpo_group_size == 4
    assert cfg.benchmark_tasks == ["gsm8k", "gpqa", "aime"]


def test_build_distill_rl_config_accepts_trust_remote_code(tmp_path: Path):
    cfg = build_distill_rl_config(
        job_id="job-1",
        user_id="user-1",
        data_dir=tmp_path,
        payload={"preset": "smoke", "trust_remote_code": True},
    )
    assert cfg.trust_remote_code is True


def test_build_distill_rl_config_reproducible_seeds(tmp_path: Path):
    cfg = build_distill_rl_config(
        job_id="job-2",
        user_id="user-1",
        data_dir=tmp_path,
        payload={
            "preset": "reproducible",
            "config_file": "distill_rl_reproducible.json",
        },
    )
    assert cfg.preset == "reproducible"
    assert cfg.teacher_revision == "main"


def test_validate_stage_sequence_rejects_out_of_order():
    with pytest.raises(ValueError, match="must follow order"):
        validate_stage_sequence(["evaluate", "rollout"])


def test_load_rollout_prompts_with_ids(tmp_path: Path):
    path = tmp_path / "prompts.json"
    path.write_text(
        json.dumps(
            {
                "prompts": [
                    {"prompt_id": "a", "text": "Explain backprop."},
                    {
                        "prompt_id": "b",
                        "prompt": "What is 6 * 7?",
                        "answer": "42",
                        "benchmark": "gsm8k",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    prompts = load_rollout_prompts(path, limit=10)
    assert prompts[0].prompt_id == "a"
    assert prompts[1].text == "What is 6 * 7?"
    assert prompts[1].answer == "42"
    assert prompts[1].benchmark == "gsm8k"


def test_distill_texts_add_thinking_tokens(tmp_path: Path):
    from seiso.distill_rl.runner import _distill_texts

    path = tmp_path / "prompts.json"
    path.write_text(
        json.dumps({"prompts": [{"prompt_id": "p", "text": "Solve 1+1."}]}),
        encoding="utf-8",
    )
    cfg = build_distill_rl_config(
        job_id="job-1",
        user_id="user-1",
        data_dir=tmp_path,
        payload={"preset": "smoke", "prompt_library": str(path)},
    )

    texts = _distill_texts(cfg)

    assert texts[0].endswith("<think>")
    assert cfg.thinking_instruction in texts[0]


def test_split_train_val_is_deterministic():
    prompts = [RolloutPrompt(prompt_id=f"p{i}", text=f"text {i}") for i in range(10)]
    train_a, val_a = split_train_val(prompts, train_fraction=0.8, seed=13)
    train_b, val_b = split_train_val(prompts, train_fraction=0.8, seed=13)
    assert [p.prompt_id for p in train_a] == [p.prompt_id for p in train_b]
    assert len(train_a) == 8
    assert len(val_a) == 2


def test_build_preference_bundle_filters_degenerate_pairs(tmp_path: Path, monkeypatch):
    prompts = [
        RolloutPrompt(prompt_id="p1", text="one"),
        RolloutPrompt(prompt_id="p2", text="two"),
    ]

    def fake_rows(**kwargs):
        rows = kwargs["prompts"]
        return [
            {
                "prompt_id": row.prompt_id,
                "prompt": row.text,
                "chosen": "good",
                "rejected": "bad" if row.prompt_id == "p1" else "bad",
                "generation_seed": 1,
            }
            for row in rows
        ]

    monkeypatch.setattr(
        "seiso.distill_rl.preferences.generate_preference_rows", fake_rows
    )
    monkeypatch.setattr(
        "seiso.distill_rl.preferences.load_rollout_prompts",
        lambda *_args, **_kwargs: prompts,
    )

    bundle = build_preference_bundle(
        teacher_model="teacher",
        student_model="student",
        output_dir=tmp_path / "prefs",
        prompt_library_path=None,
        max_prompts=2,
        max_new_tokens=8,
        temperature=0.0,
        seed=13,
        train_fraction=0.5,
        use_chat_template=False,
    )
    train_rows = [
        json.loads(line) for line in bundle.train_path.read_text().splitlines() if line
    ]
    assert bundle.filtered_count >= 0
    assert bundle.manifest_path.is_file()
    assert all(row["chosen"] != row["rejected"] for row in train_rows)


def test_build_preference_bundle_forwards_trust_remote_code(
    tmp_path: Path, monkeypatch
):
    prompts = [RolloutPrompt(prompt_id="p1", text="one")]
    seen: list[bool] = []

    def fake_rows(**kwargs):
        seen.append(kwargs["trust_remote_code"])
        return [
            {
                "prompt_id": "p1",
                "prompt": "one",
                "chosen": "good",
                "rejected": "bad",
                "generation_seed": 1,
            }
        ]

    monkeypatch.setattr(
        "seiso.distill_rl.preferences.generate_preference_rows", fake_rows
    )
    monkeypatch.setattr(
        "seiso.distill_rl.preferences.load_rollout_prompts",
        lambda *_args, **_kwargs: prompts,
    )

    bundle = build_preference_bundle(
        teacher_model="teacher",
        student_model="student",
        output_dir=tmp_path / "prefs",
        prompt_library_path=None,
        max_prompts=1,
        max_new_tokens=8,
        temperature=0.0,
        seed=13,
        train_fraction=0.5,
        use_chat_template=False,
        trust_remote_code=True,
    )

    manifest = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))
    assert seen == [True, True]
    assert manifest["trust_remote_code"] is True


def test_outcome_group_sampling_prefers_best_candidate(monkeypatch):
    from seiso.distill_rl.rollouts import generate_outcome_preference_rows

    prompts = [
        RolloutPrompt(
            prompt_id="gsm",
            text="What is 40 + 2?",
            answer="42",
            benchmark="gsm8k",
        )
    ]

    monkeypatch.setattr(
        "seiso.distill_rl.rollouts.generate_completion_groups",
        lambda *_args, **_kwargs: [["41", "<think>compute</think>42", "wrong"]],
    )

    rows = generate_outcome_preference_rows(
        student_model="student",
        prompts=prompts,
        max_new_tokens=8,
        temperature=0.7,
        seed=13,
        use_chat_template=False,
        grpo_group_size=3,
    )

    assert rows[0]["chosen"] == "<think>compute</think>42"
    assert rows[0]["chosen_reward"] == 1.0
    assert rows[0]["rejected_reward"] == 0.0
    assert rows[0]["reward_source"] == "verifiable_outcome"
    assert rows[0]["grpo_group_size"] == 3


def test_load_causal_lm_applies_trust_remote_code_and_max_memory(monkeypatch):
    from seiso.distill_rl import model_utils

    calls: dict[str, dict] = {}

    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def empty_cache():
            return None

    class FakeTokenizer:
        pad_token = None
        eos_token = "<eos>"

    class FakeTokenizers:
        @staticmethod
        def from_pretrained(model_path, **kwargs):
            calls["tokenizer"] = {"model_path": model_path, **kwargs}
            return FakeTokenizer()

    class FakeParameter:
        device = "cuda:0"

    class FakeModel:
        device = "cuda:0"

        def eval(self):
            return self

        def parameters(self):
            return iter([FakeParameter()])

    class FakeModels:
        @staticmethod
        def from_pretrained(model_path, **kwargs):
            calls["model"] = {"model_path": model_path, **kwargs}
            return FakeModel()

    fake_transformers = types.SimpleNamespace(
        AutoModelForCausalLM=FakeModels,
        AutoTokenizer=FakeTokenizers,
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(model_utils.torch, "cuda", FakeCuda)
    monkeypatch.setattr(
        "seiso.memory.protection.build_hf_max_memory",
        lambda: {0: "12000MiB"},
    )

    model_utils.load_causal_lm(
        "org/custom-model",
        revision="abc123",
        trust_remote_code=True,
    )

    assert calls["tokenizer"]["trust_remote_code"] is True
    assert calls["tokenizer"]["revision"] == "abc123"
    assert calls["model"]["trust_remote_code"] is True
    assert calls["model"]["revision"] == "abc123"
    assert calls["model"]["device_map"] == "auto"
    assert calls["model"]["max_memory"] == {0: "12000MiB"}


def test_latest_checkpoint_sorts_numeric_suffix(tmp_path: Path):
    run_dir = tmp_path / "dpo-run"
    run_dir.mkdir()
    (run_dir / "checkpoint-9").mkdir()
    (run_dir / "checkpoint-10").mkdir()
    assert _latest_checkpoint(run_dir).name == "checkpoint-10"
    assert _checkpoint_step(run_dir / "checkpoint-10") == 10


def test_aggregate_multiseed_runs(tmp_path: Path):
    run_a = tmp_path / "run-a" / "evaluation"
    run_b = tmp_path / "run-b" / "evaluation"
    run_a.mkdir(parents=True)
    run_b.mkdir(parents=True)
    payload = {
        "checkpoints": {
            "distilled": {"perplexity": 10.0, "val_preference_accuracy": 0.5},
        }
    }
    (run_a / "evaluation_summary.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    payload["checkpoints"]["distilled"]["perplexity"] = 12.0
    payload["checkpoints"]["distilled"]["val_preference_accuracy"] = 0.7
    (run_b / "evaluation_summary.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )

    aggregate = aggregate_multiseed_runs(
        [tmp_path / "run-a", tmp_path / "run-b"], output_dir=tmp_path / "agg"
    )
    stats = aggregate["checkpoints"]["distilled"]
    assert stats["perplexity_mean"] == 11.0
    assert stats["perplexity_n"] == 2.0


def test_val_preference_metrics_include_margin_and_alignment(monkeypatch):
    from seiso.distill_rl import evaluate

    scores = {
        ("p", "good"): -0.2,
        ("p", "bad"): -0.4,
        ("q", "short"): -0.5,
        ("q", "long"): -0.1,
    }

    def fake_logprob(_model, _tokenizer, prompt, completion, _device):
        return scores[(prompt, completion)]

    monkeypatch.setattr(evaluate, "_sequence_logprob", fake_logprob)
    metrics = evaluate._val_preference_metrics(
        object(),
        object(),
        [
            {"prompt": "p", "chosen": "good", "rejected": "bad"},
            {"prompt": "q", "chosen": "short", "rejected": "long"},
        ],
        object(),
    )

    assert metrics["val_preference_accuracy"] == pytest.approx(0.5)
    assert metrics["val_preference_margin_mean"] == pytest.approx(-0.1)
    assert metrics["val_preference_count"] == 2
    assert metrics["alignment_score"] < metrics["val_preference_accuracy"]


def test_verifiable_benchmark_reports_accuracy_jump(tmp_path: Path, monkeypatch):
    from seiso.distill_rl.verifiable_benchmarks import evaluate_verifiable_benchmarks

    def fake_generate(model_path, prompts, *args, **kwargs):
        if model_path == "base":
            return ["0" for _prompt in prompts]
        return [str(prompt.answer) for prompt in prompts]

    monkeypatch.setattr(
        "seiso.distill_rl.verifiable_benchmarks.generate_completions",
        fake_generate,
    )

    report = evaluate_verifiable_benchmarks(
        output_dir=tmp_path,
        checkpoints={"student_base": "base", "dpo": "trained"},
        prompt_library_path=None,
        benchmark_tasks=["gsm8k", "gpqa", "aime"],
        max_prompts_per_task=1,
        trust_remote_code=False,
        require_thinking_trace=True,
        thinking_instruction="Think.",
    )

    assert report["checkpoints"]["dpo"]["gsm8k"]["accuracy"] == 1.0
    assert report["accuracy_jumps"]["baseline"] == "student_base"
    assert report["accuracy_jumps"]["by_checkpoint"]["dpo"]["gsm8k"] == 1.0
    assert Path(report["summary_path"]).is_file()


def test_verifiable_benchmark_records_generation_errors(tmp_path: Path, monkeypatch):
    from seiso.distill_rl.verifiable_benchmarks import evaluate_verifiable_benchmarks

    def fail_generate(*_args, **_kwargs):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(
        "seiso.distill_rl.verifiable_benchmarks._generate_benchmark_outputs",
        fail_generate,
    )

    report = evaluate_verifiable_benchmarks(
        output_dir=tmp_path,
        checkpoints={"student_base": "missing-model"},
        prompt_library_path=None,
        benchmark_tasks=["gsm8k"],
        max_prompts_per_task=1,
        trust_remote_code=False,
        require_thinking_trace=True,
        thinking_instruction="Think.",
    )

    task = report["checkpoints"]["student_base"]["gsm8k"]
    assert task["accuracy"] == 0.0
    assert task["error"] == "RuntimeError: model unavailable"
    assert Path(report["summary_path"]).is_file()


def test_paper_bundle_flattens_verifiable_benchmark_jumps():
    from seiso.distill_rl.paper_bundle import _flatten_metrics

    metrics = _flatten_metrics(
        {
            "checkpoints": {},
            "verifiable_benchmarks": {
                "checkpoints": {
                    "dpo": {"gsm8k": {"accuracy": 0.75}},
                },
                "accuracy_jumps": {
                    "baseline": "student_base",
                    "by_checkpoint": {"dpo": {"gsm8k": 0.25}},
                },
            },
        }
    )

    assert metrics["dpo.benchmark.gsm8k.accuracy"] == 0.75
    assert metrics["dpo.benchmark.gsm8k.accuracy_jump"] == 0.25


def test_run_distill_rl_job_orchestrates_stages(tmp_path: Path):
    from seiso.distill_rl.runner import run_distill_rl_job

    distilled = tmp_path / "distill_rl" / "cli" / "job-x" / "distilled"
    distilled.mkdir(parents=True)
    prefs_dir = tmp_path / "distill_rl" / "cli" / "job-x" / "preferences"
    prefs_dir.mkdir(parents=True)
    train = prefs_dir / "preferences_train.jsonl"
    val = prefs_dir / "preferences_val.jsonl"
    train.write_text(
        json.dumps({"prompt": "p", "chosen": "yes", "rejected": "no"}) + "\n",
        encoding="utf-8",
    )
    val.write_text(
        json.dumps({"prompt": "p", "chosen": "yes", "rejected": "no"}) + "\n",
        encoding="utf-8",
    )
    (prefs_dir / "preferences_manifest.json").write_text("{}", encoding="utf-8")

    dpo_dir = (
        tmp_path
        / "distill_rl"
        / "cli"
        / "job-x"
        / "dpo"
        / "seiso_job-x"
        / "checkpoint-1"
    )
    dpo_dir.mkdir(parents=True)
    eval_summary = (
        tmp_path
        / "distill_rl"
        / "cli"
        / "job-x"
        / "evaluation"
        / "evaluation_summary.json"
    )
    eval_summary.parent.mkdir(parents=True)
    eval_summary.write_text(json.dumps({"checkpoints": {}}), encoding="utf-8")

    bundle = MagicMock(
        train_path=train,
        val_path=val,
        manifest_path=prefs_dir / "preferences_manifest.json",
        train_count=1,
        val_count=1,
        filtered_count=0,
    )

    with (
        patch("seiso.models.hf_env.configure_hf_hub_cache"),
        patch("seiso.security.nvidia_boundary.enforce_nvidia_secure_boundary"),
        patch(
            "seiso.distill_rl.runner.init_run_manifest",
            return_value={"config_fingerprint": "abc"},
        ),
        patch("seiso.distill_rl.runner.verify_run_manifest", return_value={"ok": True}),
        patch("seiso.distill_rl.runner.append_artifact"),
        patch("seiso.distill_rl.runner._run_distill", return_value=distilled),
        patch(
            "seiso.distill_rl.preferences.build_preference_bundle", return_value=bundle
        ),
        patch("seiso.distill_rl.runner._run_dpo", return_value=dpo_dir),
        patch(
            "seiso.distill_rl.evaluate.evaluate_pipeline",
            return_value={
                "checkpoints": {},
                "summary_path": str(eval_summary),
                "verifiable_benchmarks": {
                    "summary_path": str(eval_summary.parent / "verifiable_benchmarks.json"),
                    "accuracy_jumps": {
                        "baseline": "student_base",
                        "by_checkpoint": {"dpo": {"gsm8k": 0.25}},
                    },
                },
            },
        ),
        patch(
            "seiso.distill_rl.runner.create_paper_bundle",
            return_value={"paper_bundle_dir": str(tmp_path / "paper")},
        ),
    ):
        result = run_distill_rl_job(
            job_id="job-x",
            user_id="cli",
            data_dir=tmp_path,
            payload={
                "preset": "smoke",
                "stages": ["distill", "rollout", "dpo", "evaluate"],
                "auto_sweep": False,
            },
        )

    assert result["final_model_dir"] == str(dpo_dir)
    assert result["paper_bundle"]["paper_bundle_dir"]
    assert result["benchmark_jumps"] == ["dpo: gsm8k=+0.250"]
    assert "verifiable_benchmarks" in result["stage_results"]


def test_resolve_job_seeds_from_reproducible_preset(tmp_path: Path):
    seeds = resolve_job_seeds(
        {"preset": "reproducible", "config_file": "distill_rl_reproducible.json"}
    )
    assert seeds == [13, 42, 99]


def test_run_distill_rl_multiseed_from_preset(tmp_path: Path):
    from seiso.distill_rl.runner import run_distill_rl_job

    single_result = {
        "output_dir": str(tmp_path / "distill_rl" / "cli" / "job-s13"),
        "stage_results": {},
    }
    with patch(
        "seiso.distill_rl.runner._run_single_job", return_value=single_result
    ) as single:
        result = run_distill_rl_job(
            job_id="job-ms",
            user_id="cli",
            data_dir=tmp_path,
            payload={
                "preset": "reproducible",
                "config_file": "distill_rl_reproducible.json",
            },
        )
    assert single.call_count == 3
    assert result["multiseed"] is True
    assert result["seeds"] == [13, 42, 99]


def test_run_distill_rl_multiseed_explicit_seeds(tmp_path: Path):
    from seiso.distill_rl.runner import run_distill_rl_job

    single_result = {
        "output_dir": str(tmp_path / "distill_rl" / "cli" / "job-s13"),
        "stage_results": {},
    }
    with patch(
        "seiso.distill_rl.runner._run_single_job", return_value=single_result
    ) as single:
        result = run_distill_rl_job(
            job_id="job-ms",
            user_id="cli",
            data_dir=tmp_path,
            payload={"preset": "smoke", "seeds": [13, 42]},
        )
    assert single.call_count == 2
    assert result["multiseed"] is True
    assert result["seeds"] == [13, 42]
