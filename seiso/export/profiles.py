"""Export profiles — preset format bundles for LoRA, full fine-tune, and GGUF."""

from __future__ import annotations

from pathlib import Path

from seiso.compat import StrEnum
from seiso.export.formats import ExportFormat
from seiso.io.jsonl import read_json_file


class ExportProfile(StrEnum):
    """Named export bundles applied after training or on demand."""

    LORA_ADAPTER = "lora_adapter"
    LORA_BUNDLE = "lora_bundle"
    FULL_FINETUNE = "full_finetune"
    FULL_BUNDLE = "full_bundle"
    INFERENCE = "inference"
    GGUF_ONLY = "gguf_only"
    HUB_READY = "hub_ready"


_PROFILE_FORMATS: dict[ExportProfile, list[ExportFormat]] = {
    ExportProfile.LORA_ADAPTER: [ExportFormat.LORA],
    ExportProfile.LORA_BUNDLE: [
        ExportFormat.LORA,
        ExportFormat.MERGED,
        ExportFormat.GGUF,
    ],
    ExportProfile.FULL_FINETUNE: [ExportFormat.FULL],
    ExportProfile.FULL_BUNDLE: [ExportFormat.FULL, ExportFormat.GGUF],
    ExportProfile.INFERENCE: [ExportFormat.MERGED, ExportFormat.GGUF],
    ExportProfile.GGUF_ONLY: [ExportFormat.GGUF],
    ExportProfile.HUB_READY: [ExportFormat.MERGED],
}

_DEFAULT_GGUF_QUANTS: dict[ExportProfile, list[str]] = {
    ExportProfile.LORA_BUNDLE: ["q4_k_m", "q8_0", "f16"],
    ExportProfile.FULL_BUNDLE: ["q4_k_m", "q5_k_m", "q8_0", "f16"],
    ExportProfile.INFERENCE: ["q4_k_m", "q8_0"],
    ExportProfile.GGUF_ONLY: ["q2_k", "q3_k_m", "q4_k_m", "q5_k_m", "q8_0", "f16"],
}


def resolve_profile(name: str | ExportProfile) -> ExportProfile:
    try:
        return ExportProfile(name)
    except ValueError as exc:
        valid = ", ".join(p.value for p in ExportProfile)
        raise ValueError(f"Unknown export profile {name!r}; choose from: {valid}") from exc


def formats_for_profile(profile: ExportProfile) -> list[ExportFormat]:
    return list(_PROFILE_FORMATS[profile])


def default_gguf_quants(profile: ExportProfile) -> list[str]:
    return list(_DEFAULT_GGUF_QUANTS.get(profile, ["q4_k_m", "q8_0"]))


def detect_checkpoint_kind(checkpoint: Path) -> str:
    """Return 'lora', 'full', or 'unknown'."""
    has_adapter_config = (checkpoint / "adapter_config.json").exists()
    has_adapter_weights = (checkpoint / "adapter_model.safetensors").is_file() or (
        checkpoint / "adapter_model.bin"
    ).is_file()
    if has_adapter_config:
        return "lora"
    # Adapter weights without adapter_config still mean LoRA-only (EXP-02-R).
    # Do not treat bare config.json as a merged full checkpoint in that case.
    if has_adapter_weights:
        return "lora"
    if (checkpoint / "config.json").exists():
        return "full"
    manifest = read_json_file(checkpoint / "seiso_manifest.json", default={})
    if isinstance(manifest, dict):
        method = str(manifest.get("method", "")).lower()
        if method == "lora":
            return "lora"
        if method == "full":
            return "full"
        if method == "slime":
            return "lora" if manifest.get("adapter") == "lora" else "full"
        if method == "nemo_rl":
            return "lora" if manifest.get("adapter") == "lora" else "full"
    return "unknown"


def suggest_profile(checkpoint: Path, *, method: str | None = None) -> ExportProfile:
    """Pick a sensible default export profile from checkpoint contents."""
    kind = method or detect_checkpoint_kind(checkpoint)
    if kind == "lora":
        return ExportProfile.LORA_ADAPTER
    if kind == "full":
        return ExportProfile.FULL_BUNDLE
    return ExportProfile.INFERENCE


def resolve_formats(
    *,
    formats: list[str] | list[ExportFormat] | None = None,
    profile: str | ExportProfile | None = None,
    checkpoint: Path | None = None,
) -> list[ExportFormat]:
    """Resolve explicit formats or a profile into an ordered format list."""
    if profile:
        prof = resolve_profile(profile)
        resolved = formats_for_profile(prof)
    elif formats:
        resolved = [ExportFormat(str(f).strip().lower()) for f in formats]
    elif checkpoint is not None:
        resolved = formats_for_profile(suggest_profile(checkpoint))
    else:
        resolved = [ExportFormat.MERGED]

    # Deduplicate while preserving order
    seen: set[ExportFormat] = set()
    out: list[ExportFormat] = []
    for fmt in resolved:
        if fmt not in seen:
            seen.add(fmt)
            out.append(fmt)
    return out
