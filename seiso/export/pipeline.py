"""Export pipeline — precheck, profile resolution, and post-training auto-export."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from seiso.export.formats import ExportFormat, ExportOptions, export_checkpoint
from seiso.export.hub_precheck import HubPrecheckResult, precheck_hub_export
from seiso.export.model_card import HubModelMetadata, metadata_from_manifest
from seiso.export.profiles import (
    ExportProfile,
    default_gguf_quants,
    detect_checkpoint_kind,
    formats_for_profile,
    resolve_formats,
    resolve_profile,
    suggest_profile,
)


@dataclass
class AutoExportConfig:
    """Options for automatic export after training or RL jobs."""

    formats: list[str] | None = None
    profile: str | ExportProfile | None = None
    gguf_quantizations: list[str] | None = None
    hub_repo: str | None = None
    hub_token: str | None = None
    hub_metadata: HubModelMetadata | None = None


@dataclass
class ExportPlan:
    """Resolved export plan returned by prepare_export."""

    checkpoint: Path
    output_dir: Path
    formats: list[ExportFormat]
    gguf_quantizations: list[str]
    hub_repo: str | None = None
    hub_metadata: HubModelMetadata | None = None
    checkpoint_kind: str = "unknown"
    profile: str | None = None
    precheck: HubPrecheckResult | None = None
    warnings: list[str] = field(default_factory=list)


def prepare_export(
    *,
    checkpoint: Path,
    output_dir: Path,
    formats: list[str] | None = None,
    profile: str | ExportProfile | None = None,
    gguf_quantizations: list[str] | None = None,
    hub_repo: str | None = None,
    hub_token: str | None = None,
    hub_metadata: HubModelMetadata | None = None,
    on_log: Callable[[str], None] | None = None,
) -> ExportPlan:
    """Resolve formats and run Hub precheck before any heavy export work."""
    kind = detect_checkpoint_kind(checkpoint)
    prof = resolve_profile(profile) if profile else None
    resolved_formats = resolve_formats(formats=formats, profile=profile, checkpoint=checkpoint)
    quants = list(gguf_quantizations or (default_gguf_quants(prof) if prof else ["q4_k_m", "q8_0"]))

    meta = hub_metadata
    if meta and (checkpoint / "seiso_manifest.json").is_file():
        meta = metadata_from_manifest(meta, checkpoint / "seiso_manifest.json")
        if not meta.finetune_type and kind == "lora":
            meta.finetune_type = "lora"
        elif not meta.finetune_type and kind == "full":
            meta.finetune_type = "full"
        meta.export_formats = [f.value for f in resolved_formats]
        if ExportFormat.GGUF in resolved_formats:
            meta.quantizations = quants

    plan = ExportPlan(
        checkpoint=checkpoint,
        output_dir=output_dir,
        formats=resolved_formats,
        gguf_quantizations=quants,
        hub_repo=hub_repo,
        hub_metadata=meta,
        checkpoint_kind=kind,
        profile=prof.value if prof else None,
    )

    if hub_repo and hub_token:
        plan.precheck = precheck_hub_export(
            repo_id=hub_repo,
            token=hub_token,
            metadata=meta,
            formats=[f.value for f in resolved_formats],
            on_log=on_log,
        )
        plan.warnings.extend(plan.precheck.warnings)

    return plan


def run_export_plan(
    plan: ExportPlan,
    *,
    hub_token: str | None = None,
    sandbox_root: Path | None = None,
    on_log: Callable[[str], None] | None = None,
    require_precheck_ok: bool = True,
) -> dict[str, Path]:
    """Execute a prepared export plan."""
    if plan.precheck and require_precheck_ok and not plan.precheck.ok:
        from seiso.export.hub_precheck import assert_hub_precheck_ok

        assert_hub_precheck_ok(plan.precheck)

    options = ExportOptions(
        checkpoint=plan.checkpoint,
        output_dir=plan.output_dir,
        formats=plan.formats,
        gguf_quantizations=plan.gguf_quantizations,
        hub_repo=plan.hub_repo,
        hub_token=hub_token,
        hub_metadata=plan.hub_metadata,
        sandbox_root=sandbox_root,
        skip_hub_precheck=plan.precheck is not None and plan.precheck.ok,
    )
    return export_checkpoint(options, on_log=on_log)


def auto_export_after_training(
    checkpoint: Path,
    output_dir: Path,
    config: AutoExportConfig | dict[str, Any],
    *,
    sandbox_root: Path | None = None,
    on_log: Callable[[str], None] | None = None,
) -> dict[str, Path]:
    """Export immediately after training using profile or explicit formats."""
    if isinstance(config, dict):
        hub_meta_raw = config.get("hub_metadata")
        hub_metadata = HubModelMetadata(**hub_meta_raw) if hub_meta_raw else None
        auto = AutoExportConfig(
            formats=config.get("formats"),
            profile=config.get("profile"),
            gguf_quantizations=config.get("gguf_quantizations"),
            hub_repo=config.get("hub_repo"),
            hub_token=config.get("hub_token"),
            hub_metadata=hub_metadata,
        )
    else:
        auto = config

    profile = auto.profile
    if not profile and not auto.formats:
        profile = suggest_profile(checkpoint).value

    plan = prepare_export(
        checkpoint=checkpoint,
        output_dir=output_dir,
        formats=auto.formats,
        profile=profile,
        gguf_quantizations=auto.gguf_quantizations,
        hub_repo=auto.hub_repo,
        hub_token=auto.hub_token,
        hub_metadata=auto.hub_metadata,
        on_log=on_log,
    )
    return run_export_plan(
        plan,
        hub_token=auto.hub_token,
        sandbox_root=sandbox_root,
        on_log=on_log,
    )


def profile_catalog() -> list[dict[str, Any]]:
    """Return export profiles for API/UI."""
    return [
        {
            "id": p.value,
            "formats": [f.value for f in formats_for_profile(p)],
            "default_gguf_quants": default_gguf_quants(p),
        }
        for p in ExportProfile
    ]
