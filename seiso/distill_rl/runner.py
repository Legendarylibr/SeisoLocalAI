"""Orchestrate teacher distillation and preference-based RL (DPO)."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

from seiso.distill_rl.config import (
    DistillRLConfig,
    build_distill_rl_config,
    resolve_job_seeds,
)
from seiso.distill_rl.manifest import (
    append_artifact,
    init_run_manifest,
    verify_run_manifest,
)
from seiso.distill_rl.multiseed import aggregate_multiseed_runs
from seiso.distill_rl.paper_bundle import create_paper_bundle
from seiso.distill_rl.sweep import (
    SharedStageContext,
    apply_best_sweep_overrides,
    auto_sweep_enabled,
    run_auto_hyperparameter_sweep,
)


def run_distill_rl_job(
    *,
    job_id: str,
    user_id: str,
    data_dir: Path,
    payload: dict[str, Any],
    on_log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run distill → rollout → DPO → evaluate with research artifacts."""
    seeds = resolve_job_seeds(payload)
    if seeds:
        return _run_multiseed_job(
            job_id=job_id,
            user_id=user_id,
            data_dir=data_dir,
            payload=payload,
            seeds=[int(s) for s in seeds],
            on_log=on_log,
        )
    return _run_single_job(
        job_id=job_id,
        user_id=user_id,
        data_dir=data_dir,
        payload=payload,
        on_log=on_log,
    )


def _run_multiseed_job(
    *,
    job_id: str,
    user_id: str,
    data_dir: Path,
    payload: dict[str, Any],
    seeds: list[int],
    on_log: Callable[[str], None] | None,
) -> dict[str, Any]:
    from seiso.security import safe_join

    parent = safe_join(data_dir, "distill_rl", user_id, f"{job_id}-multiseed")
    parent.mkdir(parents=True, exist_ok=True)
    run_dirs: list[Path] = []
    seed_results: list[dict[str, Any]] = []

    for seed in seeds:
        sub_payload = {**payload, "seed": seed}
        sub_job_id = f"{job_id}-s{seed}"
        if on_log:
            on_log(f"Multi-seed run seed={seed}")
        result = _run_single_job(
            job_id=sub_job_id,
            user_id=user_id,
            data_dir=data_dir,
            payload=sub_payload,
            on_log=on_log,
        )
        seed_results.append(result)
        run_dirs.append(Path(result["output_dir"]))

    aggregate = aggregate_multiseed_runs(run_dirs, output_dir=parent)
    bundle = create_paper_bundle(
        output_root=parent,
        run_name=f"multiseed_{job_id[:8]}",
        config={"seeds": seeds, "payload": payload},
        stage_results={"runs": [r.get("output_dir") for r in seed_results]},
        evaluation=aggregate,
        manifest=None,
    )
    return {
        "multiseed": True,
        "seeds": seeds,
        "output_dir": str(parent),
        "seed_results": seed_results,
        "aggregate": aggregate,
        "paper_bundle": bundle,
    }


def _run_single_job(
    *,
    job_id: str,
    user_id: str,
    data_dir: Path,
    payload: dict[str, Any],
    on_log: Callable[[str], None] | None,
) -> dict[str, Any]:
    from seiso.models.hf_env import configure_hf_hub_cache
    from seiso.security.nvidia_boundary import enforce_nvidia_secure_boundary

    configure_hf_hub_cache(data_dir)
    enforce_nvidia_secure_boundary(context="training")

    config = build_distill_rl_config(
        job_id=job_id,
        user_id=user_id,
        data_dir=data_dir,
        payload=payload,
    )

    def _log(msg: str) -> None:
        if on_log:
            on_log(msg)

    _log(
        f"Distill-RL run preset={config.preset} stages={','.join(config.stages)} "
        f"teacher={config.teacher_model} student={config.student_model} seed={config.seed}"
    )
    config.output_root.mkdir(parents=True, exist_ok=True)
    _write_effective_config(config)
    manifest = init_run_manifest(config)

    shared = _run_shared_stages(config, on_log=on_log)

    sweep_result: dict[str, Any] | None = None
    if auto_sweep_enabled(payload) and "dpo" in config.stages:
        _log("Phase: auto hyperparameter sweep (DPO)")
        sweep_result = run_auto_hyperparameter_sweep(
            config,
            payload=payload,
            shared=shared,
            on_log=on_log,
            run_dpo_fn=_run_dpo,
        )
        config = apply_best_sweep_overrides(
            config, sweep_result.get("best_overrides") or {}
        )

    stage_results: dict[str, Any] = dict(shared.stage_results)
    distilled_dir = shared.distilled_dir
    dpo_dir: Path | None = None
    evaluation: dict[str, Any] | None = None

    if "dpo" in config.stages:
        _log("Phase: dpo (preference RL after distillation)")
        train_path = config.preferences_train_path
        if not train_path.is_file():
            raise FileNotFoundError(f"Preference train dataset missing: {train_path}")
        dpo_dir = _run_dpo(
            config, model_dir=distilled_dir, preferences_path=train_path, on_log=on_log
        )
        stage_results["dpo"] = str(dpo_dir)
        append_artifact(config.output_root, stage="dpo", artifact_path=dpo_dir)

    if "evaluate" in config.stages:
        _log("Phase: evaluate (PPL + val preference accuracy)")
        from seiso.distill_rl.evaluate import evaluate_pipeline

        checkpoints: dict[str, Path | str] = {"distilled": distilled_dir}
        if dpo_dir is not None:
            checkpoints["dpo"] = dpo_dir
        student_path = Path(config.student_model).expanduser()
        if student_path.is_dir():
            checkpoints["student_base"] = student_path.resolve()
        if config.evaluate_teacher:
            teacher_path = Path(config.teacher_model).expanduser()
            if teacher_path.is_dir():
                checkpoints["teacher"] = teacher_path.resolve()
            else:
                checkpoints["teacher"] = config.teacher_model
        val_path = config.preferences_val_path
        if not val_path.is_file():
            val_path = config.preferences_dir / "preferences_val.jsonl"
        eval_library = config.prompt_library_path
        if eval_library is None:
            from seiso.distill_rl.grounded_data import grounded_prompts_path

            grounded = grounded_prompts_path(config)
            if grounded.is_file():
                eval_library = grounded
        evaluation = evaluate_pipeline(
            output_dir=config.evaluation_dir,
            checkpoints=checkpoints,
            val_preferences_path=val_path,
            prompt_library_path=eval_library,
            eval_max_prompts=config.eval_max_prompts,
            trust_remote_code=config.trust_remote_code,
            use_chat_template=bool(config.use_chat_template),
            benchmark_verifiable=config.benchmark_verifiable,
            benchmark_tasks=config.benchmark_tasks,
            require_thinking_trace=config.require_thinking_trace,
            thinking_instruction=config.thinking_instruction,
            on_log=on_log,
        )
        stage_results["evaluation"] = evaluation.get("summary_path")
        benchmark_report = evaluation.get("verifiable_benchmarks")
        if isinstance(benchmark_report, dict) and benchmark_report.get("summary_path"):
            stage_results["verifiable_benchmarks"] = benchmark_report["summary_path"]
        append_artifact(
            config.output_root,
            stage="evaluate",
            artifact_path=Path(str(evaluation["summary_path"])),
        )
        if isinstance(benchmark_report, dict) and benchmark_report.get("summary_path"):
            append_artifact(
                config.output_root,
                stage="evaluate",
                artifact_path=Path(str(benchmark_report["summary_path"])),
                role="benchmark",
            )

    manifest_report = verify_run_manifest(config.output_root)
    if isinstance(manifest_report, dict) and manifest_report.get("ok") is False:
        raise RuntimeError(
            f"Distill-RL manifest verification failed: {manifest_report}"
        )
    try:
        from seiso.research.nostr import maybe_auto_attest

        nostr_report = maybe_auto_attest(config.output_root / "manifest.json")
        if nostr_report and nostr_report.get("ok"):
            _log(f"Nostr attestation: event_id={nostr_report.get('event_id')}")
        elif nostr_report and nostr_report.get("error"):
            _log(f"Nostr attestation skipped: {nostr_report.get('error')}")
    except Exception as exc:  # pragma: no cover - defensive
        _log(f"Nostr attestation skipped: {exc}")
    paper_bundle = create_paper_bundle(
        output_root=config.output_root,
        run_name=f"seiso_{config.job_id[:8]}",
        config=config.model_dump(mode="json"),
        stage_results=stage_results,
        evaluation=evaluation,
        manifest=manifest,
    )
    stage_results["paper_bundle"] = paper_bundle.get("paper_bundle_dir")

    final_model_dir = stage_results.get("dpo") or str(distilled_dir)
    _log("Distill-RL pipeline complete")
    benchmark_jumps: list[str] = []
    if isinstance(evaluation, dict):
        from seiso.distill_rl.verifiable_benchmarks import summarize_accuracy_jumps

        benchmark_jumps = summarize_accuracy_jumps(
            evaluation.get("verifiable_benchmarks")
        )

    result: dict[str, Any] = {
        "output_dir": str(config.output_root),
        "output_root": str(config.output_root),
        "run_dir": str(config.output_root),
        "model_dir": final_model_dir,
        "preset": config.preset,
        "seed": config.seed,
        "stages": config.stages,
        "stage_results": stage_results,
        "distilled_dir": str(distilled_dir),
        "evaluation": evaluation,
        "manifest": manifest_report,
        "paper_bundle": paper_bundle,
        "final_model_dir": final_model_dir,
        "auto_sweep": auto_sweep_enabled(payload),
        "benchmark_jumps": benchmark_jumps,
    }
    if sweep_result is not None:
        result["sweep"] = sweep_result
    return result


def _run_shared_stages(
    config: DistillRLConfig,
    *,
    on_log: Callable[[str], None] | None,
) -> SharedStageContext:
    stage_results: dict[str, Any] = {}
    distilled_dir = _resolve_policy_model_dir(config)

    if "distill" in config.stages:
        if on_log:
            on_log("Phase: distill (teacher logits → student)")
        distilled_dir = _run_distill(config, on_log=on_log)
        stage_results["distilled"] = str(distilled_dir)
        append_artifact(
            config.output_root, stage="distill", artifact_path=distilled_dir
        )
    elif not distilled_dir.is_dir():
        raise FileNotFoundError(
            f"Distilled student checkpoint missing: {distilled_dir}. "
            "Run the distill stage or set distilled_path to an existing checkpoint."
        )

    if "rollout" in config.stages:
        from seiso.distill_rl.config import allow_tiny_rl
        from seiso.distill_rl.grounded_data import materialize_distill_grounded_prompts
        from seiso.distill_rl.preferences import build_preference_bundle

        source = str(config.preference_source)
        if source == "teacher_style":
            library_path = config.prompt_library_path
            if library_path is None:
                raise ValueError(
                    "preference_source=teacher_style requires prompt_library"
                )
            if on_log:
                on_log(f"Phase: rollout (teacher_style; library={library_path})")
            min_grounded = None
        else:
            if on_log:
                on_log(
                    f"Phase: rollout (preference_source={source}; "
                    f"data_gen_count={config.data_gen_count})"
                )
            library_path = materialize_distill_grounded_prompts(config, on_log=on_log)
            # Match materialize floors: smoke preset and SEISO_ALLOW_TINY_RL.
            min_grounded = 1 if allow_tiny_rl(preset=config.preset) else None
        bundle = build_preference_bundle(
            teacher_model=config.teacher_model,
            student_model=str(distilled_dir),
            output_dir=config.preferences_dir,
            prompt_library_path=library_path,
            max_prompts=config.rollout_max_prompts,
            max_new_tokens=config.rollout_max_new_tokens,
            temperature=config.rollout_temperature,
            seed=config.seed,
            train_fraction=config.train_val_fraction,
            use_chat_template=bool(config.use_chat_template),
            teacher_revision=config.teacher_revision,
            student_revision=config.student_revision,
            trust_remote_code=config.trust_remote_code,
            require_thinking_trace=config.require_thinking_trace,
            thinking_instruction=config.thinking_instruction,
            # Normalized on DistillRLConfig from preference_source.
            verifiable_outcome_rewards=bool(config.verifiable_outcome_rewards),
            grpo_group_size=config.grpo_group_size,
            min_grounded_prompts=min_grounded,
            on_log=on_log,
        )
        stage_results["preferences_train"] = str(bundle.train_path)
        stage_results["preferences_val"] = str(bundle.val_path)
        stage_results["preferences_manifest"] = str(bundle.manifest_path)
        append_artifact(
            config.output_root, stage="rollout", artifact_path=bundle.manifest_path
        )
        append_artifact(
            config.output_root,
            stage="rollout",
            artifact_path=bundle.train_path,
            role="train",
        )
        append_artifact(
            config.output_root,
            stage="rollout",
            artifact_path=bundle.val_path,
            role="val",
        )
        config.preferences_path.write_text(
            bundle.train_path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )

    return SharedStageContext(distilled_dir=distilled_dir, stage_results=stage_results)


def _resolve_policy_model_dir(config: DistillRLConfig) -> Path:
    if config.distilled_path is not None:
        path = config.distilled_path.expanduser().resolve()
        if path.is_dir():
            return path
        raise FileNotFoundError(f"distilled_path is not a directory: {path}")

    if config.distilled_dir.is_dir():
        return config.distilled_dir

    student_path = Path(config.student_model).expanduser()
    if student_path.is_dir() and "distill" not in config.stages:
        return student_path.resolve()

    return config.distilled_dir


def _write_effective_config(config: DistillRLConfig) -> None:
    path = config.output_root / "distill_rl_config.json"
    path.write_text(
        json.dumps(config.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8"
    )


def _distill_texts(config: DistillRLConfig) -> list[str]:
    from seiso.distill_rl.grounded_data import materialize_distill_grounded_prompts
    from seiso.distill_rl.outcome import format_thinking_prompt
    from seiso.distill_rl.prompts import load_rollout_prompts, prompt_texts

    limit = config.max_train_samples or config.rollout_max_prompts
    source = str(config.preference_source)
    if source == "teacher_style" and config.prompt_library_path is not None:
        prompts = load_rollout_prompts(config.prompt_library_path, limit=limit or 0)
    elif source == "grounded_library" and config.prompt_library_path is not None:
        # Prefer shared materialize (copies/normalizes) so distill==rollout corpus.
        library = materialize_distill_grounded_prompts(config)
        prompts = load_rollout_prompts(library, limit=limit or 0)
    else:
        library = materialize_distill_grounded_prompts(config)
        prompts = load_rollout_prompts(library, limit=limit or 0)
    texts = prompt_texts(prompts)
    if config.require_thinking_trace:
        return [
            format_thinking_prompt(text, config.thinking_instruction) for text in texts
        ]
    return texts


def _run_distill(
    config: DistillRLConfig, *, on_log: Callable[[str], None] | None
) -> Path:
    from contextlib import nullcontext

    from seiso.compress.bootstrap import require_codellama_compress
    from seiso.distill_rl.distill_corpus import override_distill_corpus

    require_codellama_compress()
    from seiso.codellama_compress.config import (
        DatasetConfig,
        DistillConfig,
        merge_dataclass,
    )
    from seiso.codellama_compress.distill import run_distillation
    from seiso.codellama_compress.replay import apply_global_seeds

    if config.deterministic:
        apply_global_seeds(config.seed)

    dataset_cfg = merge_dataclass(
        DatasetConfig(),
        {"seed": config.seed, "max_train_samples": config.max_train_samples},
    )
    distill_cfg = merge_dataclass(
        DistillConfig(),
        {
            "teacher_model": config.teacher_model,
            "student_model": config.student_model,
            "steps": config.distill_steps,
            "alpha": config.distill_alpha,
            "temperature": config.distill_temperature,
        },
    )

    out_dir = config.distilled_dir
    if on_log:
        on_log(f"Distilling for {distill_cfg.steps} steps → {out_dir}")

    corpus_ctx = (
        override_distill_corpus(_distill_texts(config))
        if config.align_distill_with_prompts
        else nullcontext()
    )

    with corpus_ctx:
        run_distillation(
            run_dir=config.output_root,
            out_dir=out_dir,
            dataset_cfg=dataset_cfg,
            cfg=distill_cfg,
            seed=config.seed,
        )
    return out_dir


def _checkpoint_step(path: Path) -> int:
    if not path.name.startswith("checkpoint-"):
        return -1
    suffix = path.name.removeprefix("checkpoint-")
    try:
        return int(suffix)
    except ValueError:
        return -1


def _latest_checkpoint(run_dir: Path) -> Path | None:
    checkpoints = [path for path in run_dir.glob("checkpoint-*") if path.is_dir()]
    if not checkpoints:
        return None
    return max(checkpoints, key=_checkpoint_step)


def _run_dpo(
    config: DistillRLConfig,
    *,
    model_dir: Path,
    preferences_path: Path,
    on_log: Callable[[str], None] | None,
) -> Path:
    from seiso.distill_rl.dpo.config import DPOSettings
    from seiso.distill_rl.dpo.dpo_trainer import DPOTrainer
    from seiso.distill_rl.dpo.preference_data import load_preference_dataset

    settings = DPOSettings(
        sft_model_path=str(model_dir),
        output_dir=str(config.dpo_output_dir),
        run_name=f"seiso_{config.job_id[:8]}",
        beta=config.dpo_beta,
        average_log_prob=config.dpo_average_log_prob,
        learning_rate=config.dpo_learning_rate,
        num_epochs=config.dpo_epochs,
        per_device_train_batch_size=config.dpo_batch_size,
        gradient_accumulation_steps=config.dpo_gradient_accumulation_steps,
        max_grad_norm=config.dpo_max_grad_norm,
        warmup_ratio=config.dpo_warmup_ratio,
        weight_decay=config.dpo_weight_decay,
        preference_dataset_path=str(preferences_path),
        save_steps=config.dpo_save_steps,
        seed=config.seed,
        use_lora=config.dpo_use_lora,
        use_qlora=config.dpo_use_qlora,
        use_chat_template=bool(config.use_chat_template),
        trust_remote_code=config.trust_remote_code,
        gradient_checkpointing=True,
    )

    examples = load_preference_dataset(preferences_path)
    if config.dpo_max_steps is not None:
        settings.max_steps = int(config.dpo_max_steps)
        micro_batches = max(
            1, math.ceil(len(examples) / settings.per_device_train_batch_size)
        )
        optimizer_steps_per_epoch = max(
            1,
            math.ceil(micro_batches / settings.gradient_accumulation_steps),
        )
        # Keep epochs as an upper bound; max_steps is the hard optimizer cap.
        settings.num_epochs = max(
            1, math.ceil(config.dpo_max_steps / optimizer_steps_per_epoch)
        )

    if on_log:
        on_log(
            f"DPO on {len(examples)} train preferences "
            f"(beta={settings.beta}, lr={settings.learning_rate:g}, "
            f"effective_batch={settings.per_device_train_batch_size * settings.gradient_accumulation_steps}, "
            f"epochs={settings.num_epochs}, lora={settings.use_lora}, "
            f"chat_template={settings.use_chat_template})"
        )

    trainer = DPOTrainer(settings)
    trainer.train(examples, shuffle=True)

    run_dir = Path(settings.output_dir) / settings.run_name
    latest = _latest_checkpoint(run_dir)
    return latest if latest is not None else Path(run_dir)
