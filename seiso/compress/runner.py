"""Run bundled LLM compression pipeline stages."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from seiso.compress.bootstrap import require_codellama_compress
from seiso.compress.config_builder import build_pipeline_config


def _pipeline_fingerprint(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "preset": cfg["preset"],
        "stages": cfg["stages"],
        "dataset": cfg["dataset"],
        "distill": cfg["distill"],
        "finetune": cfg["finetune"],
        "prune": cfg["prune"],
        "determinism": cfg["determinism"],
    }


def _trust_remote_code(cfg: dict[str, Any]) -> bool:
    return bool(
        getattr(cfg.get("distill"), "trust_remote_code", False)
        or getattr(cfg.get("finetune"), "trust_remote_code", False)
    )


def _assert_full_model_dir(path: Path, stage: str) -> None:
    """Refuse LoRA-only / empty dirs when skipping distill into prune/finetune."""
    if not path.is_dir():
        raise FileNotFoundError(f"Expected model directory for {stage}: {path}")
    has_config = (path / "config.json").is_file()
    has_adapter = (path / "adapter_config.json").is_file()
    if has_adapter and not has_config:
        raise ValueError(
            f"model_dir for stage {stage!r} is a LoRA adapter only; "
            "merge to a full HF checkpoint (with config.json) first, or run distill."
        )
    if not has_config:
        raise ValueError(
            f"model_dir for stage {stage!r} must contain config.json "
            "(full Hugging Face model directory)."
        )


def _resolve_model_dir(cfg: dict[str, Any], run_dir: Path, stage: str) -> Path:
    if cfg.get("model_dir") and stage == cfg["stages"][0]:
        path = Path(cfg["model_dir"])
        _assert_full_model_dir(path, stage)
        return path
    pruned = run_dir / "pruned"
    distilled = run_dir / "distilled"
    # Prefer pruned when present; otherwise finetune from distilled.
    if pruned.is_dir():
        finetune_in = pruned
    elif distilled.is_dir():
        finetune_in = distilled
    else:
        finetune_in = pruned
    stage_inputs = {
        "distill": None,
        "prune": distilled,
        "finetune": finetune_in,
        "evaluate": _latest_model_dir(run_dir),
        "export": _latest_model_dir(run_dir),
        "quantize_gptq": _latest_model_dir(run_dir),
        "quantize_awq": _latest_model_dir(run_dir),
    }
    resolved = stage_inputs.get(stage)
    if resolved is None:
        raise ValueError(f"No input model for stage {stage}")
    if not resolved.is_dir():
        raise FileNotFoundError(f"Expected model directory for {stage}: {resolved}")
    return resolved


def _latest_model_dir(run_dir: Path) -> Path:
    for name in ("quantized-awq", "quantized-gptq", "finetuned", "pruned", "distilled"):
        candidate = run_dir / name
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"No model artifacts found under {run_dir}")


def run_compress_job(
    *,
    job_id: str,
    user_id: str,
    data_dir: Path,
    payload: dict[str, Any],
    on_log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Execute distill → prune → finetune → evaluate → export (configurable)."""
    require_codellama_compress()
    from seiso.codellama_compress.io import (
        assert_disk_budget,
        new_run_dir,
        save_effective_config,
        write_env_report,
    )
    from seiso.codellama_compress.replay import (
        append_artifact_record,
        apply_global_seeds,
        content_fingerprint,
        derive_run_id,
        init_manifest,
        verify_manifest,
    )

    cfg = build_pipeline_config(
        job_id=job_id, user_id=user_id, data_dir=data_dir, payload=payload
    )

    def _log(msg: str) -> None:
        if on_log:
            on_log(msg)

    det = cfg["determinism"]
    if det.deterministic:
        apply_global_seeds(det.seed)

    pipe_fp = _pipeline_fingerprint(cfg)
    pipeline_hash = content_fingerprint(pipe_fp)
    run_id = payload.get("run_id")
    if run_id is None and det.hash_run_id:
        run_id = derive_run_id(pipeline_hash)

    out_root = Path(cfg["output_root"])
    if cfg.get("min_free_gb") is not None:
        assert_disk_budget(root=out_root, min_free_gb=float(cfg["min_free_gb"]))

    run_dir = new_run_dir(out_root / "runs", run_id=run_id)
    save_effective_config(run_dir, {**cfg, "pipeline_fingerprint": pipe_fp})
    if cfg.get("env_report"):
        write_env_report(run_dir)
    init_manifest(
        run_dir,
        config_fingerprint=pipeline_hash,
        pipeline_fingerprint=pipe_fp,
        determinism=det,
        effective_config=cfg,
        stage=cfg["stages"][0],
    )

    _log(
        f"Compression run: {run_dir.name} preset={cfg['preset']} stages={','.join(cfg['stages'])}"
    )

    stage_results: dict[str, Any] = {}
    for stage in cfg["stages"]:
        _log(f"Phase: {stage}")
        if stage == "distill":
            from seiso.codellama_compress.distill import run_distillation

            out_dir = run_dir / "distilled"
            run_distillation(
                run_dir=run_dir,
                out_dir=out_dir,
                dataset_cfg=cfg["dataset"],
                cfg=cfg["distill"],
                seed=det.seed,
            )
            append_artifact_record(
                run_dir, stage="distill", artifact_path=out_dir, role="output"
            )
            stage_results["distilled"] = str(out_dir)

        elif stage == "prune":
            from seiso.codellama_compress.prune import run_mlp_mask_prune

            in_dir = _resolve_model_dir(cfg, run_dir, "prune")
            out_dir = run_dir / "pruned"
            run_mlp_mask_prune(
                in_model_dir=in_dir,
                out_dir=out_dir,
                ratio=cfg["prune"]["ratio"],
                method=cfg["prune"]["method"],
                seed=det.seed,
                trust_remote_code=_trust_remote_code(cfg),
            )
            append_artifact_record(
                run_dir, stage="prune", artifact_path=out_dir, role="output"
            )
            stage_results["pruned"] = str(out_dir)

        elif stage == "finetune":
            from seiso.codellama_compress.finetune import run_finetune

            in_dir = _resolve_model_dir(cfg, run_dir, "finetune")
            out_dir = run_dir / "finetuned"
            run_finetune(
                run_dir=run_dir,
                in_model_dir=in_dir,
                out_dir=out_dir,
                dataset_cfg=cfg["dataset"],
                cfg=cfg["finetune"],
                seed=det.seed,
            )
            append_artifact_record(
                run_dir, stage="finetune", artifact_path=out_dir, role="output"
            )
            stage_results["finetuned"] = str(out_dir)

        elif stage == "evaluate":
            from seiso.codellama_compress.evaluate import evaluate_into_run_dir

            model_dir = _resolve_model_dir(cfg, run_dir, "evaluate")
            result = evaluate_into_run_dir(
                run_dir=run_dir,
                model_dir=model_dir,
                trust_remote_code=_trust_remote_code(cfg),
            )
            stage_results["evaluate"] = (
                result.to_dict() if hasattr(result, "to_dict") else result
            )
            _log(str(result))

        elif stage == "export":
            from seiso.codellama_compress.export import write_export_bundle

            model_dir = _resolve_model_dir(cfg, run_dir, "export")
            export_dir = run_dir / "export_bundle"
            write_export_bundle(
                model_dir=model_dir,
                out_dir=export_dir,
                model_name=cfg["export"]["model_name"],
                port=cfg["export"]["port"],
            )
            append_artifact_record(
                run_dir, stage="export", artifact_path=export_dir, role="output"
            )
            stage_results["export_bundle"] = str(export_dir)
            _log(f"Export bundle: {export_dir}")

        elif stage == "quantize_gptq":
            from seiso.codellama_compress.quantize_gptq import run_gptq_quantization

            in_dir = _resolve_model_dir(cfg, run_dir, "quantize_gptq")
            out_dir = run_dir / "quantized-gptq"
            run_gptq_quantization(
                run_dir=run_dir,
                in_model_dir=in_dir,
                out_dir=out_dir,
                dataset_cfg=cfg["dataset"],
                cfg=cfg["gptq"],
                trust_remote_code=_trust_remote_code(cfg),
            )
            append_artifact_record(
                run_dir, stage="quantize_gptq", artifact_path=out_dir, role="output"
            )
            stage_results["quantized_gptq"] = str(out_dir)

        elif stage == "quantize_awq":
            from seiso.codellama_compress.quantize_awq import run_awq_quantization

            in_dir = _resolve_model_dir(cfg, run_dir, "quantize_awq")
            out_dir = run_dir / "quantized-awq"
            run_awq_quantization(
                run_dir=run_dir,
                in_model_dir=in_dir,
                out_dir=out_dir,
                dataset_cfg=cfg["dataset"],
                cfg=cfg["awq"],
                trust_remote_code=_trust_remote_code(cfg),
            )
            append_artifact_record(
                run_dir, stage="quantize_awq", artifact_path=out_dir, role="output"
            )
            stage_results["quantized_awq"] = str(out_dir)

        else:
            raise ValueError(f"Unhandled stage: {stage}")

    if cfg.get("max_run_dir_gb") is not None:
        assert_disk_budget(
            root=out_root,
            max_dir_gb=float(cfg["max_run_dir_gb"]),
            dir_path=run_dir,
        )

    manifest_report = verify_manifest(run_dir)
    _log(f"Manifest verify: ok={manifest_report.get('ok')}")
    if not manifest_report.get("ok"):
        raise RuntimeError(
            f"Compression manifest verification failed: {manifest_report}"
        )
    try:
        from seiso.research.nostr import maybe_auto_attest

        nostr_report = maybe_auto_attest(run_dir / "manifest.json")
        if nostr_report and nostr_report.get("ok"):
            _log(f"Nostr attestation: event_id={nostr_report.get('event_id')}")
        elif nostr_report and nostr_report.get("error"):
            _log(f"Nostr attestation skipped: {nostr_report.get('error')}")
    except Exception as exc:  # pragma: no cover - defensive
        _log(f"Nostr attestation skipped: {exc}")
    _log("Compression pipeline complete")

    return {
        "run_dir": str(run_dir),
        "output_root": str(out_root),
        "run_id": run_dir.name,
        "stages": cfg["stages"],
        "stage_results": stage_results,
        # Prefer the latest pipeline artifact (quant beats earlier finetune/distill).
        "model_dir": stage_results.get("quantized_awq")
        or stage_results.get("quantized_gptq")
        or stage_results.get("finetuned")
        or stage_results.get("pruned")
        or stage_results.get("distilled")
        or cfg.get("model_dir"),
        "manifest": manifest_report,
    }
