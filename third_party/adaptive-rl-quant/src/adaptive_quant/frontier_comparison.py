"""Compare hosted frontier model outputs against local llama.cpp generations."""

from __future__ import annotations

import random
import sys
from typing import Any

from adaptive_quant.backends.llama_cpp import require_llama_cpp_paths, run_llama_cpp_completion
from adaptive_quant.configuration import FrameworkConfig
from adaptive_quant.frontier_reference import (
    FrontierReferenceClient,
    LocalCompletion,
    frontier_record_from_completion,
    length_ratio,
    normalize_frontier_reference_payload,
    save_frontier_reference,
    word_overlap_score,
)
from adaptive_quant.hardware import host_aware_hardware_profiles
from adaptive_quant.logging_utils import read_json, write_json
from adaptive_quant.math_utils import mean
from adaptive_quant.prompts import PromptLibrary, PromptSample
from adaptive_quant.types import HardwareType


def select_comparison_prompts(config: FrameworkConfig) -> list[PromptSample]:
    library = PromptLibrary()
    limit = int(config.frontier_comparison_prompt_limit)
    prompts = list(library.prompts)
    if limit <= 0 or limit >= len(prompts):
        return prompts
    rng = random.Random(int(config.seed) + 31337)
    return rng.sample(prompts, limit)


def score_frontier_reference(
    config: FrameworkConfig,
    *,
    prompts: list[PromptSample] | None = None,
    output_path: str | None = None,
) -> dict[str, Any]:
    if not (config.frontier_enabled or config.frontier_auto_compare_in_pipeline):
        raise ValueError("frontier_enabled must be true to score frontier reference outputs")
    selected = prompts or select_comparison_prompts(config)
    client = FrontierReferenceClient(config)
    records: dict[str, dict[str, Any]] = {}
    for prompt in selected:
        completion = client.complete(prompt)
        records[prompt.prompt_id] = frontier_record_from_completion(completion)
    target = output_path or config.frontier_reference_path()
    meta = {
        "model": config.frontier_model,
        "api_base": config.frontier_api_base,
        "prompt_count": len(records),
    }
    save_frontier_reference(target, records, meta=meta)
    return {"path": target, "prompt_count": len(records), "records": records}


def _load_local_reference_map(path: str) -> dict[str, str]:
    payload = read_json(path, label="Local reference file")
    texts: dict[str, str] = {}
    if isinstance(payload, dict) and isinstance(payload.get("prompts"), dict):
        source = payload["prompts"]
    elif isinstance(payload, dict):
        source = payload
    else:
        raise TypeError("Local reference file must be a JSON object")
    for prompt_id, value in source.items():
        if prompt_id == "meta" or not isinstance(prompt_id, str):
            continue
        if isinstance(value, dict):
            text = value.get("response_text")
            if isinstance(text, str):
                texts[prompt_id] = text
        elif isinstance(value, str):
            texts[prompt_id] = value
    return texts


def _local_completion_for_prompt(
    config: FrameworkConfig,
    prompt: PromptSample,
    *,
    local_reference: dict[str, str],
) -> LocalCompletion | None:
    cached = local_reference.get(prompt.prompt_id)
    if cached is not None:
        return LocalCompletion(
            prompt_id=prompt.prompt_id,
            response_text=cached,
            latency_ms=None,
            source="local_reference_file",
        )
    try:
        llama_cpp_binary, llama_cpp_model = require_llama_cpp_paths(config)
    except (FileNotFoundError, ValueError):
        return None
    profiles = host_aware_hardware_profiles(None)
    gpu_profile = profiles[HardwareType.GPU]
    metrics, generated = run_llama_cpp_completion(
        config,
        llama_cpp_binary=llama_cpp_binary,
        llama_cpp_model=llama_cpp_model,
        prompt_text=prompt.text,
        ngl=int(gpu_profile.ngl),
    )
    return LocalCompletion(
        prompt_id=prompt.prompt_id,
        response_text=generated,
        latency_ms=float(metrics.get("latency_ms", metrics.get("latency_ms_per_token", 0.0))),
        source="llama_cpp",
    )


def compare_frontier_to_local(
    config: FrameworkConfig,
    *,
    prompts: list[PromptSample] | None = None,
    reference_path: str | None = None,
    refresh_reference: bool = False,
) -> dict[str, Any]:
    if not (config.frontier_enabled or config.frontier_auto_compare_in_pipeline):
        raise ValueError(
            "frontier_enabled or frontier_auto_compare_in_pipeline must be true for comparison"
        )
    selected = prompts or select_comparison_prompts(config)
    ref_path = reference_path or config.frontier_reference_path()
    if refresh_reference or not _reference_file_exists(ref_path):
        score_frontier_reference(config, prompts=selected, output_path=ref_path)
    reference_payload = read_json(ref_path, label="Frontier reference file")
    reference_records = normalize_frontier_reference_payload(reference_payload)

    local_reference: dict[str, str] = {}
    if config.frontier_local_reference_path:
        local_reference = _load_local_reference_map(config.frontier_local_reference_path)

    rows: list[dict[str, Any]] = []
    skipped_missing_reference = 0
    for prompt in selected:
        ref = reference_records.get(prompt.prompt_id)
        if ref is None:
            skipped_missing_reference += 1
            continue
        frontier_text = str(ref.get("response_text") or "")
        local = _local_completion_for_prompt(config, prompt, local_reference=local_reference)
        overlap = None
        len_ratio = None
        local_text = ""
        local_source = "unavailable"
        local_latency_ms = None
        if local is not None:
            local_text = local.response_text
            local_source = local.source
            local_latency_ms = local.latency_ms
            overlap = word_overlap_score(frontier_text, local_text)
            len_ratio = length_ratio(len(frontier_text), len(local_text))
        rows.append(
            {
                "prompt_id": prompt.prompt_id,
                "prompt_domain": prompt.domain,
                "frontier_model": ref.get("model", config.frontier_model),
                "frontier_latency_ms": ref.get("latency_ms"),
                "frontier_response_excerpt": _excerpt(frontier_text),
                "local_source": local_source,
                "local_latency_ms": local_latency_ms,
                "local_response_excerpt": _excerpt(local_text),
                "reference_overlap": overlap,
                "length_ratio": len_ratio,
            }
        )

    overlap_values = [
        float(row["reference_overlap"]) for row in rows if row["reference_overlap"] is not None
    ]
    if skipped_missing_reference:
        print(
            f"[frontier] skipped {skipped_missing_reference} prompt(s) with no reference entry",
            file=sys.stderr,
        )
    summary = {
        "run_name": config.run_name,
        "frontier_model": config.frontier_model,
        "local_backend": config.backend,
        "comparison_metric": config.frontier_comparison_metric,
        "reference_path": ref_path,
        "prompts_compared": len(rows),
        "prompts_skipped_missing_reference": skipped_missing_reference,
        "mean_reference_overlap": mean(overlap_values) if overlap_values else 0.0,
        "rows": rows,
    }
    comparison_path = config.frontier_comparison_path()
    write_json(comparison_path, summary)

    if config.frontier_write_external_quality_sidecar and overlap_values:
        sidecar_path = _external_quality_sidecar_path(config)
        sidecar = {
            row["prompt_id"]: {config.frontier_comparison_metric: row["reference_overlap"]}
            for row in rows
            if row["reference_overlap"] is not None
        }
        write_json(sidecar_path, sidecar)
        summary["external_quality_sidecar"] = sidecar_path

    summary["comparison_path"] = comparison_path
    return summary


def _reference_file_exists(path: str) -> bool:
    from pathlib import Path

    return Path(path).is_file()


def _external_quality_sidecar_path(config: FrameworkConfig) -> str:
    return f"{config.benchmark_dir}/{config.run_name}_frontier_external_quality.json"


def _excerpt(text: str, *, limit: int = 240) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


__all__ = [
    "compare_frontier_to_local",
    "score_frontier_reference",
    "select_comparison_prompts",
]
