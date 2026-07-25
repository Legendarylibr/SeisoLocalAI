"""Gates for RLVR-aligned product data paths (no toy product sources)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from seiso.distill_rl.config import build_distill_rl_config
from seiso.rl_verify.synth_materialize import (
    SynthRequest,
    materialize_grounded_corpus,
)
from seiso.slime.config import SingleGpuSlimeConfig


def test_product_rejects_code_corpus_source(tmp_path: Path):
    with pytest.raises(ValueError, match="not a product training path"):
        materialize_grounded_corpus(
            tmp_path / "out.jsonl",
            SynthRequest(source="code_corpus", count=8, allow_tiny=True),  # type: ignore[arg-type]
        )


def test_grounded_library_floor_without_allow_tiny(tmp_path: Path):
    lib = tmp_path / "tiny.jsonl"
    rows = [
        {
            "prompt": [{"role": "user", "content": f"What is {i}+{i}?"}],
            "label": str(i + i),
            "reward": "numeric",
            "metadata": {"rm_type": "numeric"},
        }
        for i in range(8)
    ]
    lib.write_text(
        "".join(json.dumps(r) + "\n" for r in rows),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="need >= 256"):
        materialize_grounded_corpus(
            tmp_path / "out.jsonl",
            SynthRequest(
                source="grounded_library",
                dataset_ref=lib,
                count=8,
                allow_tiny=False,
            ),
        )


def test_grounded_library_allow_tiny_for_ci(tmp_path: Path):
    lib = tmp_path / "tiny.jsonl"
    lib.write_text(
        json.dumps(
            {
                "prompt": [{"role": "user", "content": "2+2?"}],
                "label": "4",
                "reward": "numeric",
                "metadata": {"rm_type": "numeric"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    result = materialize_grounded_corpus(
        tmp_path / "out.jsonl",
        SynthRequest(
            source="grounded_library",
            dataset_ref=lib,
            count=1,
            allow_tiny=True,
            min_verifiable=1,
            require_thinking_trace=True,
            thinking_instruction=(
                "Show your reasoning in <think>...</think>, then give the final answer."
            ),
        ),
    )
    assert result.count == 1
    content = result.rows[0]["prompt"][0]["content"]
    # Generation-time formatters prime <think>; do not bake it into the corpus.
    assert "<think>" not in content


def test_distill_fingerprint_uses_resolved_dd_endpoint(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("SEISO_ALLOW_TINY_RL", "1")
    monkeypatch.setenv("SEISO_DATA_DESIGNER_BASE_URL", "http://dd.example/v1")
    from seiso.distill_rl.config import build_distill_rl_config
    from seiso.distill_rl.grounded_data import _grounded_fingerprint

    cfg = build_distill_rl_config(
        job_id="job-dd-fp",
        user_id="user-1",
        data_dir=tmp_path,
        payload={
            "preset": "smoke",
            "preference_source": "data_designer",
            "data_gen_count": 8,
            "stages": ["rollout"],
        },
    )
    fp = _grounded_fingerprint(cfg)
    assert fp["data_designer_base_url"] == "http://dd.example/v1"


def test_data_designer_fails_without_endpoint(tmp_path: Path):
    with patch(
        "seiso.rl_verify.data_designer_gen.data_designer_available",
        return_value=True,
    ):
        with pytest.raises(RuntimeError, match="No silent localhost"):
            materialize_grounded_corpus(
                tmp_path / "out.jsonl",
                SynthRequest(
                    source="data_designer",
                    count=8,
                    endpoint=None,
                    allow_tiny=True,
                    min_verifiable=1,
                ),
            )


def test_hf_dataset_prep_keeps_verifiable_only(tmp_path: Path, monkeypatch):
    class _FakeDS(list):
        pass

    samples = _FakeDS(
        [
            {"instruction": "What is 3+4?", "output": "7"},
            {"instruction": "Say hello", "output": ""},
            {"prompt": "2*5?", "answer": "10"},
        ]
    )

    monkeypatch.setattr(
        "seiso.training.datasets.load_training_dataset",
        lambda *a, **k: samples,
    )
    monkeypatch.setattr(
        "seiso.training.preprocess.preprocess_training_dataset",
        lambda raw, **k: (raw, {"kept": len(raw)}, type("F", (), {"value": "auto"})()),
    )
    monkeypatch.setattr(
        "seiso.models.hf_env.configure_hf_hub_auth",
        lambda: None,
    )
    result = materialize_grounded_corpus(
        tmp_path / "out.jsonl",
        SynthRequest(
            source="dataset",
            dataset_ref="local/fake",
            count=10,
            allow_tiny=True,
            min_verifiable=1,
            preprocess=True,
        ),
    )
    assert result.count == 2
    assert result.meta.get("dropped_unverifiable", 0) >= 1


def test_distill_smoke_uses_fixture_grounded_library(tmp_path: Path):
    cfg = build_distill_rl_config(
        job_id="job-smoke",
        user_id="user-1",
        data_dir=tmp_path,
        payload={"preset": "smoke"},
    )
    assert cfg.preference_source == "grounded_library"
    assert cfg.prompt_library_path is not None
    assert cfg.prompt_library_path.name == "distill_verifiable_prompts.jsonl"


def test_distill_reproducible_requires_hf_dataset(tmp_path: Path):
    with pytest.raises(ValueError, match="dataset_ref"):
        build_distill_rl_config(
            job_id="job-repro",
            user_id="user-1",
            data_dir=tmp_path,
            payload={"preset": "reproducible"},
        )


def test_distill_rejects_code_corpus_preference_source(tmp_path: Path):
    with pytest.raises(ValueError, match="not a product"):
        build_distill_rl_config(
            job_id="job-x",
            user_id="user-1",
            data_dir=tmp_path,
            payload={
                "preset": "smoke",
                "preference_source": "code_corpus",
            },
        )


def test_slime_require_held_out_eval(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("SEISO_ALLOW_TINY_RL", raising=False)
    train = tmp_path / "train.jsonl"
    train.write_text("{}\n", encoding="utf-8")
    cfg = SingleGpuSlimeConfig(
        model_id="m",
        dataset=train,
        output_dir=tmp_path / "out",
        require_held_out_eval=True,
        rollouts_per_prompt=2,
    )
    with pytest.raises(ValueError, match="eval_dataset is required"):
        cfg.validate()


def test_slime_held_out_ok_when_disjoint(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("SEISO_ALLOW_TINY_RL", raising=False)
    train = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    train.write_text("{}\n", encoding="utf-8")
    eval_path.write_text("{}\n", encoding="utf-8")
    cfg = SingleGpuSlimeConfig(
        model_id="m",
        dataset=train,
        eval_dataset=eval_path,
        output_dir=tmp_path / "out",
        require_held_out_eval=True,
        rollouts_per_prompt=2,
    )
    cfg.validate()
    assert cfg.eval_dataset == eval_path


def test_slime_data_gen_source_off_fails_loud(tmp_path: Path):
    from seiso.slime.trainer import (
        _DistributedSlimeContext,
        _maybe_materialize_data_gen,
    )

    cfg = SingleGpuSlimeConfig(
        model_id="m",
        dataset=tmp_path / "train.jsonl",
        output_dir=tmp_path / "out",
        data_gen=True,
        data_gen_count=8,
        data_gen_source="off",
        require_held_out_eval=False,
        rollouts_per_prompt=2,
    )
    dist = _DistributedSlimeContext(
        enabled=False, world_size=1, rank=0, local_rank=0, device="cpu"
    )
    with (
        patch("seiso.slime.trainer._distributed_barrier"),
        patch(
            "seiso.rl_verify.data_designer_gen.data_designer_available",
            return_value=True,
        ),
        patch(
            "seiso.rl_verify.data_designer_gen.should_use_data_designer",
            return_value=True,
        ),
        pytest.raises(RuntimeError, match="data_gen_source is off"),
    ):
        _maybe_materialize_data_gen(cfg, dist)


def test_slime_hf_dataset_alone_does_not_rewrite_dataset(tmp_path: Path):
    from seiso.slime.trainer import (
        _DistributedSlimeContext,
        _maybe_materialize_data_gen,
    )

    train = tmp_path / "train.jsonl"
    train.write_text('{"prompt":"x","label":"1"}\n', encoding="utf-8")
    cfg = SingleGpuSlimeConfig(
        model_id="m",
        dataset=train,
        output_dir=tmp_path / "out",
        dataset_ref="some/hub-id",
        data_gen=False,
        require_held_out_eval=False,
        rollouts_per_prompt=2,
    )
    dist = _DistributedSlimeContext(
        enabled=False, world_size=1, rank=0, local_rank=0, device="cpu"
    )
    out = _maybe_materialize_data_gen(cfg, dist)
    assert out.dataset == train


def test_slime_hf_dataset_alone_does_not_skip_held_out_gate(
    tmp_path: Path, monkeypatch
):
    monkeypatch.delenv("SEISO_ALLOW_TINY_RL", raising=False)
    train = tmp_path / "train.jsonl"
    train.write_text("{}\n", encoding="utf-8")
    cfg = SingleGpuSlimeConfig(
        model_id="m",
        dataset=train,
        output_dir=tmp_path / "out",
        dataset_ref="org/ds",
        data_gen=False,
        require_held_out_eval=True,
        rollouts_per_prompt=2,
    )
    with pytest.raises(ValueError, match="eval_dataset is required"):
        cfg.validate()


def test_slime_orphan_data_gen_count_does_not_skip_held_out_gate(
    tmp_path: Path, monkeypatch
):
    monkeypatch.delenv("SEISO_ALLOW_TINY_RL", raising=False)
    train = tmp_path / "train.jsonl"
    train.write_text("{}\n", encoding="utf-8")
    cfg = SingleGpuSlimeConfig(
        model_id="m",
        dataset=train,
        output_dir=tmp_path / "out",
        data_gen=False,
        data_gen_count=400,
        data_gen_source="off",
        require_held_out_eval=True,
        rollouts_per_prompt=2,
    )
    with pytest.raises(ValueError, match="eval_dataset is required"):
        cfg.validate()


def test_slime_auto_without_hf_or_dd_does_not_skip_held_out_gate(
    tmp_path: Path, monkeypatch
):
    monkeypatch.delenv("SEISO_ALLOW_TINY_RL", raising=False)
    train = tmp_path / "train.jsonl"
    train.write_text("{}\n", encoding="utf-8")
    cfg = SingleGpuSlimeConfig(
        model_id="m",
        dataset=train,
        output_dir=tmp_path / "out",
        data_gen=True,
        data_gen_count=100,
        data_gen_source="auto",
        data_designer="off",
        require_held_out_eval=True,
        rollouts_per_prompt=2,
    )
    with pytest.raises(ValueError, match="eval_dataset is required"):
        cfg.validate()


def test_distill_rollout_min_grounded_honors_allow_tiny_env(
    tmp_path: Path, monkeypatch
):
    """SEISO_ALLOW_TINY_RL must lower preference floors, not only materialize."""
    monkeypatch.setenv("SEISO_ALLOW_TINY_RL", "1")
    from seiso.distill_rl.config import build_distill_rl_config
    from seiso.distill_rl.runner import _run_shared_stages

    fixture = Path("data/distill_verifiable_prompts.jsonl").resolve()
    distilled = tmp_path / "distilled"
    distilled.mkdir()
    cfg = build_distill_rl_config(
        job_id="job-tiny",
        user_id="user-1",
        data_dir=tmp_path,
        payload={
            "preset": "reproducible",
            "dataset_ref": str(fixture),
            "data_gen_count": 8,
            "stages": ["rollout"],
            "distilled_path": str(distilled),
            "seeds": [13],
        },
    )
    captured: dict = {}

    def _bundle(**kwargs):
        captured.update(kwargs)
        train = cfg.preferences_dir / "preferences_train.jsonl"
        val = cfg.preferences_dir / "preferences_val.jsonl"
        cfg.preferences_dir.mkdir(parents=True, exist_ok=True)
        train.write_text("{}\n", encoding="utf-8")
        val.write_text("{}\n", encoding="utf-8")
        manifest = cfg.preferences_dir / "preferences_manifest.json"
        manifest.write_text("{}", encoding="utf-8")
        return type(
            "B",
            (),
            {
                "train_path": train,
                "val_path": val,
                "manifest_path": manifest,
            },
        )()

    with (
        patch(
            "seiso.distill_rl.grounded_data.materialize_distill_grounded_prompts",
            return_value=fixture,
        ),
        patch(
            "seiso.distill_rl.preferences.build_preference_bundle",
            side_effect=_bundle,
        ),
        patch("seiso.distill_rl.runner.append_artifact"),
    ):
        _run_shared_stages(cfg, on_log=None)
    assert captured.get("min_grounded_prompts") == 1


def test_hf_dataset_rejects_preference_only_corpus(tmp_path: Path, monkeypatch):
    class _FakeDS(list):
        pass

    samples = _FakeDS(
        [
            {"prompt": "q1", "chosen": "a", "rejected": "b"},
            {"prompt": "q2", "chosen": "c", "rejected": "d"},
        ]
    )
    monkeypatch.setattr(
        "seiso.training.datasets.load_training_dataset",
        lambda *a, **k: samples,
    )
    monkeypatch.setattr(
        "seiso.training.preprocess.preprocess_training_dataset",
        lambda raw, **k: (raw, {"kept": len(raw)}, type("F", (), {"value": "auto"})()),
    )
    monkeypatch.setattr("seiso.models.hf_env.configure_hf_hub_auth", lambda: None)
    with pytest.raises(ValueError, match="preference-only"):
        materialize_grounded_corpus(
            tmp_path / "out.jsonl",
            SynthRequest(
                source="dataset",
                dataset_ref="local/prefs",
                count=10,
                allow_tiny=True,
                min_verifiable=1,
            ),
        )


def test_hf_dataset_scans_past_preference_head(tmp_path: Path, monkeypatch):
    """Pref-only rows before verifiable labels must not early-exit as preference-only."""

    class _FakeDS(list):
        pass

    # Longer than prior empty_scan_cap (1024) to catch regressions.
    samples = _FakeDS(
        [{"prompt": f"pref-{i}", "chosen": "a", "rejected": "b"} for i in range(1100)]
        + [{"prompt": "What is 2+2?", "answer": "4"}]
    )
    monkeypatch.setattr(
        "seiso.training.datasets.load_training_dataset",
        lambda *a, **k: samples,
    )
    monkeypatch.setattr(
        "seiso.training.preprocess.preprocess_training_dataset",
        lambda raw, **k: (raw, {"kept": len(raw)}, type("F", (), {"value": "auto"})()),
    )
    monkeypatch.setattr("seiso.models.hf_env.configure_hf_hub_auth", lambda: None)
    result = materialize_grounded_corpus(
        tmp_path / "out.jsonl",
        SynthRequest(
            source="dataset",
            dataset_ref="org/mixed-prefs-then-math",
            count=1,
            allow_tiny=True,
            min_verifiable=1,
        ),
    )
    assert result.count == 1
    assert "2+2" in result.path.read_text(encoding="utf-8")


def test_hf_auth_skipped_for_local_path(tmp_path: Path, monkeypatch):
    local = tmp_path / "verifiable.jsonl"
    local.write_text(
        json.dumps({"prompt": "What is 1+1?", "answer": "2"}) + "\n",
        encoding="utf-8",
    )

    class _FakeDS(list):
        pass

    samples = _FakeDS([{"prompt": "What is 1+1?", "answer": "2"}])
    calls: list[str] = []

    def _auth():
        calls.append("auth")
        return None

    monkeypatch.setattr(
        "seiso.training.datasets.load_training_dataset",
        lambda *a, **k: samples,
    )
    monkeypatch.setattr(
        "seiso.training.preprocess.preprocess_training_dataset",
        lambda raw, **k: (raw, {"kept": len(raw)}, type("F", (), {"value": "auto"})()),
    )
    monkeypatch.setattr("seiso.models.hf_env.configure_hf_hub_auth", _auth)
    result = materialize_grounded_corpus(
        tmp_path / "out.jsonl",
        SynthRequest(
            source="dataset",
            dataset_ref=str(local),
            count=1,
            allow_tiny=True,
            min_verifiable=1,
        ),
    )
    assert result.count == 1
    assert calls == []


def test_hf_auth_called_for_hub_id(tmp_path: Path, monkeypatch):
    class _FakeDS(list):
        pass

    samples = _FakeDS([{"prompt": "What is 3+3?", "answer": "6"}])
    calls: list[str] = []
    load_kwargs: dict = {}

    def _auth():
        calls.append("auth")
        return None

    def _load(*a, **k):
        load_kwargs.update(k)
        return samples

    monkeypatch.setattr("seiso.training.datasets.load_training_dataset", _load)
    monkeypatch.setattr(
        "seiso.training.preprocess.preprocess_training_dataset",
        lambda raw, **k: (raw, {"kept": len(raw)}, type("F", (), {"value": "auto"})()),
    )
    monkeypatch.setattr("seiso.models.hf_env.configure_hf_hub_auth", _auth)
    result = materialize_grounded_corpus(
        tmp_path / "out.jsonl",
        SynthRequest(
            source="dataset",
            dataset_ref="org/public-math",
            count=1,
            allow_tiny=True,
            min_verifiable=1,
            revision="refs/pr/1",
        ),
    )
    assert result.count == 1
    assert calls == ["auth"]
    assert load_kwargs.get("revision") == "refs/pr/1"


def test_slime_held_out_split_keeps_train_floor(tmp_path: Path, monkeypatch):
    """Auto-split must not leave train below the verifiable floor."""
    monkeypatch.delenv("SEISO_ALLOW_TINY_RL", raising=False)
    from seiso.rl_verify.synth_materialize import SynthResult
    from seiso.slime.trainer import (
        _DistributedSlimeContext,
        _maybe_materialize_data_gen,
    )

    cfg = SingleGpuSlimeConfig(
        model_id="m",
        dataset=tmp_path / "placeholder.jsonl",
        output_dir=tmp_path / "run",
        data_gen=True,
        data_gen_count=256,
        data_gen_source="dataset",
        dataset_ref="org/math",
        data_gen_filename="gen.jsonl",
        require_held_out_eval=True,
        rollouts_per_prompt=2,
    )
    dist = _DistributedSlimeContext(
        enabled=False, world_size=1, rank=0, local_rank=0, device="cpu"
    )
    rows = [
        {
            "prompt": [{"role": "user", "content": f"q{i}"}],
            "label": str(i),
            "answer": str(i),
            "reward": "numeric",
            "metadata": {"rm_type": "numeric"},
        }
        for i in range(256)
    ]

    def _materialize(out_path, request):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            "".join(json.dumps(r) + "\n" for r in rows),
            encoding="utf-8",
        )
        return SynthResult(rows=rows, source="dataset", path=out_path)

    with (
        patch(
            "seiso.rl_verify.synth_materialize.materialize_grounded_corpus",
            side_effect=_materialize,
        ),
        patch("seiso.slime.trainer._distributed_barrier"),
        pytest.raises(RuntimeError, match="auto-split left .* train rows"),
    ):
        _maybe_materialize_data_gen(cfg, dist)


def test_slime_does_not_attach_stale_held_out(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("SEISO_ALLOW_TINY_RL", "1")
    from seiso.rl_verify.synth_materialize import SynthResult
    from seiso.slime.trainer import (
        _DistributedSlimeContext,
        _maybe_materialize_data_gen,
    )

    run = tmp_path / "run"
    run.mkdir()
    stale = run / "slime_held_out_prompts.jsonl"
    stale.write_text(
        json.dumps({"prompt": "OLD_EVAL", "label": "0"}) + "\n",
        encoding="utf-8",
    )
    cfg = SingleGpuSlimeConfig(
        model_id="m",
        dataset=tmp_path / "placeholder.jsonl",
        output_dir=run,
        data_gen=True,
        data_gen_count=8,
        data_gen_source="dataset",
        dataset_ref="org/math",
        data_gen_filename="gen.jsonl",
        require_held_out_eval=False,
        rollouts_per_prompt=2,
    )
    dist = _DistributedSlimeContext(
        enabled=False, world_size=1, rank=0, local_rank=0, device="cpu"
    )
    rows = [
        {
            "prompt": [{"role": "user", "content": f"q{i}"}],
            "label": str(i),
            "answer": str(i),
            "reward": "numeric",
            "metadata": {"rm_type": "numeric"},
        }
        for i in range(8)
    ]

    def _materialize(out_path, request):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            "".join(json.dumps(r) + "\n" for r in rows),
            encoding="utf-8",
        )
        return SynthResult(rows=rows, source="dataset", path=out_path)

    with (
        patch(
            "seiso.rl_verify.synth_materialize.materialize_grounded_corpus",
            side_effect=_materialize,
        ),
        patch("seiso.slime.trainer._distributed_barrier"),
    ):
        out = _maybe_materialize_data_gen(cfg, dist)

    assert out.eval_dataset is None
    assert not stale.is_file()


def test_distill_grounded_cache_respects_fingerprint(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("SEISO_ALLOW_TINY_RL", "1")
    from seiso.distill_rl.config import build_distill_rl_config
    from seiso.distill_rl.grounded_data import (
        grounded_prompts_path,
        materialize_distill_grounded_prompts,
    )

    fixture = tmp_path / "uploads" / "user-1" / "corpus.jsonl"
    fixture.parent.mkdir(parents=True)
    fixture.write_text(
        json.dumps(
            {
                "prompt": [{"role": "user", "content": "What is 1+1?"}],
                "label": "2",
                "answer": "2",
                "reward": "numeric",
                "metadata": {"rm_type": "numeric"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    cfg = build_distill_rl_config(
        job_id="job-cache",
        user_id="user-1",
        data_dir=tmp_path,
        payload={
            "preset": "reproducible",
            "dataset_ref": str(fixture),
            "data_gen_count": 8,
            "stages": ["rollout"],
            "seeds": [13],
        },
    )
    out = grounded_prompts_path(cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "prompt": [{"role": "user", "content": "STALE"}],
                "label": "1",
                "answer": "1",
                "reward": "numeric",
                "metadata": {"rm_type": "numeric"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    calls: list[str] = []

    def _materialize(path, request):
        calls.append("materialize")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "prompt": [{"role": "user", "content": "FRESH"}],
                    "label": "2",
                    "answer": "2",
                    "reward": "numeric",
                    "metadata": {"rm_type": "numeric"},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return type("R", (), {"count": 1, "path": path, "meta": {}})()

    with patch(
        "seiso.distill_rl.grounded_data.materialize_grounded_corpus",
        side_effect=_materialize,
    ):
        path1 = materialize_distill_grounded_prompts(cfg)
        path2 = materialize_distill_grounded_prompts(cfg)
    assert calls == ["materialize"]
    assert "FRESH" in path1.read_text(encoding="utf-8")
    assert path2 == path1

    # In-place edit of the local corpus must invalidate the fingerprint cache.
    fixture.write_text(
        fixture.read_text(encoding="utf-8")
        + json.dumps({"prompt": "extra edit", "answer": "1"})
        + "\n",
        encoding="utf-8",
    )
    with patch(
        "seiso.distill_rl.grounded_data.materialize_grounded_corpus",
        side_effect=_materialize,
    ):
        materialize_distill_grounded_prompts(cfg)
    assert calls == ["materialize", "materialize"]


def test_distill_grounded_cache_invalidates_when_tiny_allow_flips(
    tmp_path: Path, monkeypatch
):
    from seiso.distill_rl.config import build_distill_rl_config
    from seiso.distill_rl.grounded_data import materialize_distill_grounded_prompts

    fixture = tmp_path / "uploads" / "user-1" / "corpus.jsonl"
    fixture.parent.mkdir(parents=True)
    fixture.write_text(
        "".join(
            json.dumps(
                {
                    "prompt": [{"role": "user", "content": f"q{i}"}],
                    "label": str(i),
                    "answer": str(i),
                    "reward": "numeric",
                    "metadata": {"rm_type": "numeric"},
                }
            )
            + "\n"
            for i in range(8)
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SEISO_ALLOW_TINY_RL", "1")
    cfg_tiny = build_distill_rl_config(
        job_id="job-floor",
        user_id="user-1",
        data_dir=tmp_path,
        payload={
            "preset": "reproducible",
            "dataset_ref": str(fixture),
            "data_gen_count": 256,
            "stages": ["rollout", "evaluate"],
            "seeds": [13],
        },
    )
    calls: list[str] = []

    def _materialize(path, request):
        calls.append("materialize")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(fixture.read_text(encoding="utf-8"), encoding="utf-8")
        return type("R", (), {"count": 8, "path": path, "meta": {}})()

    with patch(
        "seiso.distill_rl.grounded_data.materialize_grounded_corpus",
        side_effect=_materialize,
    ):
        materialize_distill_grounded_prompts(cfg_tiny)
    assert calls == ["materialize"]

    monkeypatch.delenv("SEISO_ALLOW_TINY_RL", raising=False)
    cfg_product = build_distill_rl_config(
        job_id="job-floor",
        user_id="user-1",
        data_dir=tmp_path,
        payload={
            "preset": "reproducible",
            "dataset_ref": str(fixture),
            "data_gen_count": 256,
            "stages": ["rollout", "evaluate"],
            "seeds": [13],
        },
    )
    with patch(
        "seiso.distill_rl.grounded_data.materialize_grounded_corpus",
        side_effect=_materialize,
    ):
        # allow_tiny flipped → fingerprint miss → must rematerialize (not reuse).
        materialize_distill_grounded_prompts(cfg_product)
    assert calls == ["materialize", "materialize"]


def test_slime_hf_materialize_auto_answer_field_not_default_label(tmp_path: Path):
    from seiso.slime.trainer import (
        _DistributedSlimeContext,
        _maybe_materialize_data_gen,
    )

    cfg = SingleGpuSlimeConfig(
        model_id="m",
        dataset=tmp_path / "placeholder.jsonl",
        output_dir=tmp_path / "run",
        data_gen=True,
        data_gen_count=4,
        data_gen_source="dataset",
        dataset_ref="org/math",
        answer_field="label",  # slime default
        data_gen_filename="gen.jsonl",
        require_held_out_eval=False,
        rollouts_per_prompt=2,
    )
    dist = _DistributedSlimeContext(
        enabled=False, world_size=1, rank=0, local_rank=0, device="cpu"
    )
    captured: list[SynthRequest] = []

    def _materialize(out_path, request):
        captured.append(request)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "prompt": [{"role": "user", "content": "q"}],
                    "label": "1",
                    "answer": "1",
                    "reward": "numeric",
                    "metadata": {"rm_type": "numeric"},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return type(
            "R",
            (),
            {
                "count": 1,
                "path": out_path,
                "meta": {},
                "summary": lambda self=None: {"count": 1},
            },
        )()

    with (
        patch(
            "seiso.rl_verify.synth_materialize.materialize_grounded_corpus",
            side_effect=_materialize,
        ),
        patch("seiso.slime.trainer._distributed_barrier"),
        patch("seiso.slime.config.allow_tiny_rl", return_value=True),
    ):
        _maybe_materialize_data_gen(cfg, dist)
    assert captured and captured[0].answer_field is None


def test_split_train_val_allows_fraction_one():
    from seiso.distill_rl.prompts import RolloutPrompt, split_train_val

    prompts = [
        RolloutPrompt(prompt_id="a", text="a"),
        RolloutPrompt(prompt_id="b", text="b"),
        RolloutPrompt(prompt_id="c", text="c"),
    ]
    train, val = split_train_val(prompts, train_fraction=1.0, seed=0)
    assert len(train) + len(val) == 3
    assert len(val) >= 1
    assert len(train) >= 1


@pytest.mark.asyncio
async def test_forge_distill_rejects_cross_user_hf_dataset_in_config_file(
    tmp_path: Path,
):
    """config_file must not smuggle a victim-scoped local hf_dataset past Forge."""
    from fastapi import HTTPException

    from forge.api.routes.distill_rl import (
        DistillRLStartRequest,
        _prepare_distill_rl_config,
    )
    from forge.config import ForgeSettings

    attacker = "user-a"
    victim = "user-b"
    victim_file = tmp_path / "uploads" / victim / "secret.jsonl"
    victim_file.parent.mkdir(parents=True)
    victim_file.write_text('{"prompt":"x","answer":"1"}\n', encoding="utf-8")

    cfg_path = tmp_path / "uploads" / attacker / "job.json"
    cfg_path.parent.mkdir(parents=True)
    cfg_path.write_text(
        json.dumps(
            {
                "preset": "smoke",
                "hf_dataset": str(victim_file),
                "preference_source": "dataset",
                "data_gen_count": 8,
            }
        ),
        encoding="utf-8",
    )

    settings = ForgeSettings(data_dir=tmp_path)
    body = DistillRLStartRequest(preset="smoke", config_file=str(cfg_path))
    with pytest.raises(HTTPException) as exc_info:
        await _prepare_distill_rl_config(body, db=None, user_id=attacker, settings=settings)  # type: ignore[arg-type]
    assert exc_info.value.status_code == 403


def test_forge_rl_quant_rejects_cross_user_model_in_config_file(tmp_path: Path):
    """config_file must not smuggle a victim-scoped llama_cpp_model past Forge."""
    from fastapi import HTTPException

    from forge.api.routes.rl_quant import (
        RLQuantStartRequest,
        _prepare_rl_quant_config,
    )
    from forge.config import ForgeSettings

    attacker = "user-a"
    victim = "user-b"
    victim_gguf = tmp_path / "exports" / victim / "job1" / "model.gguf"
    victim_gguf.parent.mkdir(parents=True)
    victim_gguf.write_bytes(b"GGUF")

    cfg_path = tmp_path / "uploads" / attacker / "rl.json"
    cfg_path.parent.mkdir(parents=True)
    cfg_path.write_text(
        json.dumps(
            {
                "preset": "minimal",
                "llama_cpp_model": str(victim_gguf),
            }
        ),
        encoding="utf-8",
    )

    settings = ForgeSettings(data_dir=tmp_path)
    body = RLQuantStartRequest(preset="minimal", config_file=str(cfg_path))
    with pytest.raises(HTTPException) as exc_info:
        _prepare_rl_quant_config(body, attacker, settings)
    assert exc_info.value.status_code == 403


def test_distill_rejects_outcome_false_for_grounded_sources(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("SEISO_ALLOW_TINY_RL", "1")
    from seiso.distill_rl.config import build_distill_rl_config

    fixture = tmp_path / "uploads" / "user-1" / "corpus.jsonl"
    fixture.parent.mkdir(parents=True)
    fixture.write_text(
        json.dumps(
            {
                "prompt": [{"role": "user", "content": "1+1?"}],
                "label": "2",
                "answer": "2",
                "reward": "numeric",
                "metadata": {"rm_type": "numeric"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="verifiable_outcome_rewards=true"):
        build_distill_rl_config(
            job_id="job-outcome",
            user_id="user-1",
            data_dir=tmp_path,
            payload={
                "preset": "smoke",
                "stages": ["rollout"],
                "preference_source": "dataset",
                "dataset_ref": str(fixture),
                "data_gen_count": 1,
                "verifiable_outcome_rewards": False,
            },
        )


def test_distill_runner_passes_config_outcome_rewards(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("SEISO_ALLOW_TINY_RL", "1")
    from seiso.distill_rl.config import build_distill_rl_config
    from seiso.distill_rl.runner import _run_shared_stages

    fixture = tmp_path / "uploads" / "user-1" / "corpus.jsonl"
    fixture.parent.mkdir(parents=True)
    fixture.write_text(
        json.dumps(
            {
                "prompt": [{"role": "user", "content": "1+1?"}],
                "label": "2",
                "answer": "2",
                "reward": "numeric",
                "metadata": {"rm_type": "numeric"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    distilled = tmp_path / "uploads" / "user-1" / "distilled"
    distilled.mkdir(parents=True)
    cfg = build_distill_rl_config(
        job_id="job-outcome",
        user_id="user-1",
        data_dir=tmp_path,
        payload={
            "preset": "smoke",
            "stages": ["rollout"],
            "preference_source": "dataset",
            "dataset_ref": str(fixture),
            "distilled_path": str(distilled),
            "data_gen_count": 1,
            "rollout_max_prompts": 1,
        },
    )
    captured: dict[str, object] = {}

    def _fake_bundle(**kwargs):
        captured.update(kwargs)
        from types import SimpleNamespace

        train = tmp_path / "train.jsonl"
        val = tmp_path / "val.jsonl"
        manifest = tmp_path / "manifest.json"
        train.write_text("{}\n", encoding="utf-8")
        val.write_text("{}\n", encoding="utf-8")
        manifest.write_text("{}\n", encoding="utf-8")
        return SimpleNamespace(
            train_path=train,
            val_path=val,
            manifest_path=manifest,
        )

    with (
        patch(
            "seiso.distill_rl.grounded_data.materialize_distill_grounded_prompts",
            return_value=fixture,
        ),
        patch(
            "seiso.distill_rl.preferences.build_preference_bundle",
            side_effect=_fake_bundle,
        ),
        patch("seiso.distill_rl.runner.append_artifact"),
    ):
        _run_shared_stages(cfg, on_log=None)
    assert captured.get("verifiable_outcome_rewards") is True


def test_hf_dataset_rejects_cross_user_local_path(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("SEISO_ALLOW_TINY_RL", "1")
    from seiso.security import SecurityError

    victim = tmp_path / "uploads" / "user-b" / "secret.jsonl"
    victim.parent.mkdir(parents=True)
    victim.write_text(
        json.dumps(
            {
                "prompt": [{"role": "user", "content": "x"}],
                "label": "1",
                "answer": "1",
                "reward": "numeric",
                "metadata": {"rm_type": "numeric"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(SecurityError, match="user-a"):
        materialize_grounded_corpus(
            tmp_path / "out.jsonl",
            SynthRequest(
                source="dataset",
                dataset_ref=str(victim),
                count=1,
                allow_tiny=True,
                min_verifiable=1,
                sandbox_root=tmp_path,
                sandbox_user_id="user-a",
            ),
        )


def test_slime_dd_endpoint_uses_managed_vllm_url(tmp_path: Path):
    from seiso.rl_verify.data_gen import DataGenResult
    from seiso.slime.trainer import (
        _DistributedSlimeContext,
        _maybe_materialize_data_gen,
    )

    cfg = SingleGpuSlimeConfig(
        model_id="m",
        dataset=tmp_path / "placeholder.jsonl",
        output_dir=tmp_path / "run",
        data_gen=True,
        data_gen_count=4,
        data_gen_source="data_designer",
        data_designer="on",
        vllm_base_url="",
        data_gen_filename="gen.jsonl",
        require_held_out_eval=False,
        rollouts_per_prompt=2,
    )
    dist = _DistributedSlimeContext(
        enabled=False, world_size=1, rank=0, local_rank=0, device="cpu"
    )
    fake = DataGenResult(
        rows=[{"prompt": "x", "label": "1"} for _ in range(4)],
        stream_counts={"numeric": 4},
        difficulty_counts={},
        seed=0,
    )

    def _materialize(config, *, out_path, count, world_size=1):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text('{"prompt":"x","label":"1"}\n', encoding="utf-8")
        return fake

    with (
        patch(
            "seiso.rl_verify.data_designer_gen.data_designer_available",
            return_value=True,
        ),
        patch(
            "seiso.rl_verify.data_designer_gen.should_use_data_designer",
            return_value=True,
        ),
        patch(
            "seiso.slime.rollout_backend.resolve_vllm_base_url",
            return_value="http://127.0.0.1:9999",
        ),
        patch(
            "seiso.rl_verify.data_designer_gen.materialize_for_slime_config",
            side_effect=_materialize,
        ) as m_dd,
        patch("seiso.slime.trainer._distributed_barrier"),
    ):
        out_cfg = _maybe_materialize_data_gen(cfg, dist)

    assert m_dd.called
    assert out_cfg.dataset.name == "gen.jsonl"
