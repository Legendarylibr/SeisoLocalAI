"""NVIDIA NeMo Data Designer synth data for multi-GPU vLLM slime runs.

Uses `data-designer` (https://github.com/NVIDIA-NeMo/DataDesigner) with a local
OpenAI-compatible vLLM endpoint as the model provider. Gated so only multi-GPU
vLLM slime rollouts take this path; single-GPU HF and SGLang keep the deterministic
`seiso.rl_verify.data_gen` corpus.

Code-stream rows still come from Seiso's verifiable code generators (unit-test
grounded). Numeric/choice streams are paraphrased/authored via Data Designer + vLLM.
"""

from __future__ import annotations

import json
import logging
import random
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from seiso.rl_verify.data_gen import (
    DataGenConfig,
    DataGenResult,
    parse_weight_mix,
    to_slime_prompt_row,
    write_jsonl,
)
from seiso.slime_single_gpu.config import SingleGpuSlimeConfig

logger = logging.getLogger(__name__)

_MODEL_ALIAS = "seiso-vllm-sdg"
_PROVIDER_NAME = "seiso_vllm"
_STREAMS = frozenset({"numeric", "choice", "code"})
_DIFFICULTIES = frozenset({"easy", "medium", "hard"})


@dataclass(frozen=True)
class DataDesignerGenConfig:
    """Request for NeMo Data Designer → slime JSONL."""

    count: int
    seed: int
    vllm_base_url: str
    vllm_model: str
    vllm_api_key: str = "EMPTY"
    mix: str | dict[str, float] = "numeric:0.5,choice:0.2,code:0.3"
    difficulty: str | dict[str, float] = "easy:0.35,medium:0.45,hard:0.20"
    require_thinking_trace: bool = True
    thinking_instruction: str = (
        "Show your reasoning in <think>...</think>, then give the final answer."
    )
    artifact_dir: Path | None = None
    max_tokens: int = 512
    temperature: float = 0.85
    top_p: float = 0.95


def data_designer_available() -> bool:
    """Return True when the optional ``data-designer`` package is importable."""
    try:
        import data_designer.config  # noqa: F401
        from data_designer.interface import DataDesigner  # noqa: F401

        return True
    except Exception:
        return False


def normalize_data_designer_mode(value: Any) -> str:
    key = str(value if value is not None else "auto").lower().strip()
    if key in {"off", "false", "0", "no", "disable", "disabled"}:
        return "off"
    if key in {"on", "true", "1", "yes", "force", "always"}:
        return "on"
    return "auto"


def is_multigpu_vllm_run(
    config: SingleGpuSlimeConfig,
    *,
    world_size: int = 1,
) -> bool:
    """True when slime is driving multi-GPU vLLM rollouts (DDP and/or TP)."""
    from seiso.slime_single_gpu.rollout_backend import resolve_rollout_backend

    backend = resolve_rollout_backend(config, world_size=world_size)
    if backend != "vllm":
        return False
    if world_size > 1:
        return True
    tp = int(getattr(config, "vllm_tensor_parallel", 0) or 0)
    if tp > 1:
        return True
    try:
        from seiso.inference.managed_vllm import get_status

        status = get_status()
        if status.get("running") and int(status.get("tensor_parallel_size") or 1) > 1:
            return True
    except Exception:
        pass
    return False


def should_use_data_designer(
    config: SingleGpuSlimeConfig,
    *,
    world_size: int = 1,
) -> bool:
    """Gate: Data Designer synth only on multi-GPU vLLM runs (unless forced off)."""
    mode = normalize_data_designer_mode(getattr(config, "data_designer", "auto"))
    if mode == "off":
        return False
    # mode is auto or on — both require the multigpu vllm gate.
    return is_multigpu_vllm_run(config, world_size=world_size)


def ensure_openai_v1_endpoint(base_url: str) -> str:
    """Normalize host root or ``.../v1`` into an OpenAI-compatible base URL."""
    url = (base_url or "").strip().rstrip("/")
    if not url:
        raise ValueError("vllm_base_url is required for Data Designer synth data")
    lowered = url.lower()
    if not (lowered.startswith("http://") or lowered.startswith("https://")):
        raise ValueError(f"vllm_base_url must use http(s)://, got {base_url!r}")
    if lowered.endswith("/v1"):
        return url
    return f"{url}/v1"


def _thinking_suffix(cfg: DataDesignerGenConfig) -> str:
    if not cfg.require_thinking_trace:
        return ""
    return f"\n\n{cfg.thinking_instruction}"


def _category_values_weights(
    mix: Mapping[str, float],
) -> tuple[list[str], list[float]]:
    values = [k for k, w in mix.items() if w > 0]
    weights = [float(mix[k]) for k in values]
    return values, weights


def build_rl_structured_schema() -> dict[str, Any]:
    """JSON schema for Data Designer structured generation (no Pydantic required)."""
    return {
        "type": "object",
        "properties": {
            "problem": {
                "type": "string",
                "description": "Self-contained user-facing question or problem statement.",
            },
            "answer": {
                "type": "string",
                "description": "Short ground-truth answer (number, short phrase, or letter).",
            },
        },
        "required": ["problem", "answer"],
        "additionalProperties": False,
    }


def build_data_designer_columns(
    *,
    stream_mix: Mapping[str, float],
    difficulty_mix: Mapping[str, float],
    model_alias: str = _MODEL_ALIAS,
) -> list[Any]:
    """Column configs for stream/difficulty sampling + structured problem/answer."""
    import data_designer.config as dd

    # LLM only authors numeric/choice; code is blended in post from Seiso generators.
    llm_streams = {k: v for k, v in stream_mix.items() if k in {"numeric", "choice"} and v > 0}
    if not llm_streams:
        llm_streams = {"numeric": 0.7, "choice": 0.3}
    total = sum(llm_streams.values()) or 1.0
    llm_streams = {k: v / total for k, v in llm_streams.items()}

    stream_values, stream_weights = _category_values_weights(llm_streams)
    diff_values, diff_weights = _category_values_weights(difficulty_mix)

    system = (
        "You write verifiable training prompts for local RL. "
        "Return only the structured fields. Answers must be short and exact "
        "(a number, a single letter A-D, or a short token). "
        "Do not include chain-of-thought in the problem or answer fields."
    )
    prompt = (
        "Create one {{ stream }} problem at {{ difficulty }} difficulty.\n"
        "- If stream is numeric: arithmetic or short word problem with a single numeric answer.\n"
        "- If stream is choice: multiple-choice with options A-D and answer as a single letter.\n"
        "Keep the problem self-contained in one paragraph."
    )
    return [
        dd.SamplerColumnConfig(
            name="stream",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(values=stream_values, weights=stream_weights),
        ),
        dd.SamplerColumnConfig(
            name="difficulty",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(values=diff_values, weights=diff_weights),
        ),
        dd.LLMStructuredColumnConfig(
            name="item",
            model_alias=model_alias,
            prompt=prompt,
            system_prompt=system,
            output_format=build_rl_structured_schema(),
        ),
    ]


def _extract_item(record: Mapping[str, Any]) -> tuple[str, str]:
    item = record.get("item")
    if isinstance(item, str):
        try:
            item = json.loads(item)
        except json.JSONDecodeError:
            item = None
    if isinstance(item, Mapping):
        problem = str(item.get("problem") or item.get("prompt") or "").strip()
        answer = str(item.get("answer") or item.get("label") or "").strip()
        return problem, answer
    problem = str(record.get("problem") or record.get("prompt") or "").strip()
    answer = str(record.get("answer") or record.get("label") or "").strip()
    return problem, answer


def records_to_slime_rows(
    records: list[Mapping[str, Any]],
    *,
    require_thinking_trace: bool,
    thinking_instruction: str,
    source_name: str = "nvidia.data_designer",
) -> list[dict[str, Any]]:
    """Map Data Designer records into slime-compatible prompt rows."""
    rows: list[dict[str, Any]] = []
    suffix = ""
    if require_thinking_trace and thinking_instruction:
        suffix = f"\n\n{thinking_instruction}"
    for rec in records:
        problem, answer = _extract_item(rec)
        if not problem or not answer:
            continue
        stream = str(rec.get("stream") or "numeric").lower().strip()
        if stream not in {"numeric", "choice"}:
            stream = "numeric"
        difficulty = str(rec.get("difficulty") or "medium").lower().strip()
        if difficulty not in _DIFFICULTIES:
            difficulty = "medium"
        content = f"{problem.rstrip()}{suffix}"
        rows.append(
            to_slime_prompt_row(
                content,
                answer,
                rm_type=stream,
                metadata={
                    "rm_type": stream,
                    "benchmark": stream,
                    "difficulty": difficulty,
                    "source_name": source_name,
                    "generator": "nvidia.nemo.data_designer",
                },
            )
        )
    return rows


def _allocate_counts(total: int, mix: Mapping[str, float]) -> dict[str, int]:
    if total <= 0:
        return {k: 0 for k in mix}
    keys = list(mix)
    raw = [mix[k] * total for k in keys]
    counts = {k: int(v) for k, v in zip(keys, raw, strict=True)}
    # Distribute remainder by largest fractional part.
    frac = sorted(
        ((raw[i] - counts[keys[i]], keys[i]) for i in range(len(keys))),
        reverse=True,
    )
    left = total - sum(counts.values())
    for i in range(left):
        counts[frac[i % len(frac)][1]] += 1
    return counts


def _code_rows(
    *,
    count: int,
    seed: int,
    require_thinking_trace: bool,
    thinking_instruction: str,
    difficulty_mix: Mapping[str, float],
) -> list[dict[str, Any]]:
    if count <= 0:
        return []
    from seiso.rl_verify.data_gen import DataGenConfig, generate_rl_corpus

    # Reuse Seiso code stream only by setting mix code:1.0.
    cfg = DataGenConfig(
        count=count,
        seed=seed + 17,
        mix="code:1.0",
        difficulty=dict(difficulty_mix),
        require_thinking_trace=require_thinking_trace,
        thinking_instruction=thinking_instruction,
        verify_code=True,
    )
    result = generate_rl_corpus(cfg)
    for row in result.rows:
        meta = dict(row.get("metadata") or {})
        meta["source_name"] = "seiso.data_gen+code"
        meta["generator"] = "seiso.rl_verify.data_gen.code"
        row["metadata"] = meta
    return list(result.rows)


def _import_data_designer():
    try:
        import data_designer.config as dd
        from data_designer.interface import DataDesigner
    except ImportError as exc:
        raise ImportError(
            "NVIDIA NeMo Data Designer is required for multi-GPU vLLM synth data. "
            "Install with: pip install 'data-designer>=0.8.0' "
            "(or pip install -e '.[data-designer]')."
        ) from exc
    return dd, DataDesigner


def generate_with_data_designer(cfg: DataDesignerGenConfig) -> DataGenResult:
    """Run Data Designer against local vLLM and return slime-ready rows."""
    dd, DataDesigner = _import_data_designer()

    stream_mix = parse_weight_mix(
        cfg.mix,
        allowed=_STREAMS,
        default={"numeric": 0.5, "choice": 0.2, "code": 0.3},
    )
    difficulty_mix = parse_weight_mix(
        cfg.difficulty,
        allowed=_DIFFICULTIES,
        default={"easy": 0.35, "medium": 0.45, "hard": 0.20},
    )
    counts = _allocate_counts(cfg.count, stream_mix)
    n_code = int(counts.get("code", 0))
    n_llm = max(0, cfg.count - n_code)

    rows: list[dict[str, Any]] = []
    if n_llm > 0:
        endpoint = ensure_openai_v1_endpoint(cfg.vllm_base_url)
        provider = dd.ModelProvider(
            name=_PROVIDER_NAME,
            endpoint=endpoint,
            provider_type="openai",
            api_key=cfg.vllm_api_key or "EMPTY",
        )
        model_config = dd.ModelConfig(
            alias=_MODEL_ALIAS,
            model=cfg.vllm_model,
            provider=_PROVIDER_NAME,
            inference_parameters=dd.ChatCompletionInferenceParams(
                temperature=float(cfg.temperature),
                top_p=float(cfg.top_p),
                max_tokens=int(cfg.max_tokens),
            ),
            skip_health_check=True,
        )
        builder = dd.DataDesignerConfigBuilder(model_configs=[model_config])
        for column in build_data_designer_columns(
            stream_mix=stream_mix,
            difficulty_mix=difficulty_mix,
            model_alias=_MODEL_ALIAS,
        ):
            builder.add_column(column)

        artifact_dir = Path(
            cfg.artifact_dir
            if cfg.artifact_dir is not None
            else Path.cwd() / "artifacts" / "data_designer"
        )
        artifact_dir.mkdir(parents=True, exist_ok=True)
        designer = DataDesigner(
            artifact_path=artifact_dir,
            model_providers=[provider],
            auto_configure_logging=False,
        )
        # Best-effort seed for sampler reproducibility across DD versions.
        try:
            from data_designer.config.run_config import RunConfig

            designer.set_run_config(RunConfig())
        except Exception:
            pass

        logger.info(
            "Data Designer synth: %s LLM rows via vLLM endpoint %s model=%s",
            n_llm,
            endpoint,
            cfg.vllm_model,
        )
        results = designer.create(
            builder,
            num_records=n_llm,
            dataset_name=f"slime_rl_sdg_s{cfg.seed}",
        )
        try:
            frame = results.load_dataset()
            records = frame.to_dict(orient="records")
        except Exception:
            # Fallback: export JSONL then reload.
            export_path = artifact_dir / f"slime_rl_export_{cfg.seed}.jsonl"
            results.export(export_path, format="jsonl")
            records = []
            with export_path.open(encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        rows.extend(
            records_to_slime_rows(
                records,
                require_thinking_trace=cfg.require_thinking_trace,
                thinking_instruction=cfg.thinking_instruction,
            )
        )

    rows.extend(
        _code_rows(
            count=n_code,
            seed=cfg.seed,
            require_thinking_trace=cfg.require_thinking_trace,
            thinking_instruction=cfg.thinking_instruction,
            difficulty_mix=difficulty_mix,
        )
    )

    rng = random.Random(cfg.seed)
    rng.shuffle(rows)
    # Pad if LLM dropped invalid rows.
    if len(rows) < cfg.count:
        logger.warning(
            "Data Designer produced %s/%s usable rows; padding with Seiso numeric stream",
            len(rows),
            cfg.count,
        )
        pad = DataGenConfig(
            count=cfg.count - len(rows),
            seed=cfg.seed + 91,
            mix="numeric:1.0",
            difficulty=dict(difficulty_mix),
            require_thinking_trace=cfg.require_thinking_trace,
            thinking_instruction=cfg.thinking_instruction,
        )
        from seiso.rl_verify.data_gen import generate_rl_corpus

        rows.extend(generate_rl_corpus(pad).rows)
    rows = rows[: cfg.count]

    stream_counts: dict[str, int] = {}
    difficulty_counts: dict[str, int] = {}
    for row in rows:
        meta = row.get("metadata") or {}
        stream = str(meta.get("rm_type") or row.get("reward") or "unknown")
        stream_counts[stream] = stream_counts.get(stream, 0) + 1
        diff = str(meta.get("difficulty") or "unknown")
        difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1

    return DataGenResult(
        rows=rows,
        stream_counts=stream_counts,
        difficulty_counts=difficulty_counts,
        seed=cfg.seed,
    )


def materialize_data_designer_corpus(
    out_path: Path,
    cfg: DataDesignerGenConfig,
    *,
    write_manifest: bool = True,
) -> DataGenResult:
    """Generate with Data Designer and write slime JSONL (+ optional manifest)."""
    result = generate_with_data_designer(cfg)
    n = write_jsonl(out_path, result.rows)
    if write_manifest:
        manifest_path = out_path.with_name(out_path.stem + ".manifest.json")
        if out_path.suffix != ".jsonl":
            manifest_path = out_path.with_suffix(out_path.suffix + ".manifest.json")
        manifest = {
            **result.summary(),
            "path": str(out_path),
            "written": n,
            "generator": "nvidia.nemo.data_designer",
            "vllm_base_url": cfg.vllm_base_url,
            "vllm_model": cfg.vllm_model,
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return result


def materialize_for_slime_config(
    config: SingleGpuSlimeConfig,
    *,
    out_path: Path,
    count: int,
    world_size: int = 1,
) -> DataGenResult:
    """Build DataDesignerGenConfig from slime config and materialize."""
    from seiso.slime_single_gpu.rollout_backend import resolve_vllm_base_url

    base = resolve_vllm_base_url(config) or str(
        getattr(config, "vllm_base_url", "") or ""
    )
    model = str(getattr(config, "vllm_model", "") or "").strip() or config.model_id
    api_key = str(getattr(config, "vllm_api_key", "EMPTY") or "EMPTY")
    seed = int(
        config.data_gen_seed if config.data_gen_seed is not None else config.seed
    )
    artifact_dir = config.output_dir / "data_designer_artifacts"
    cfg = DataDesignerGenConfig(
        count=count,
        seed=seed,
        vllm_base_url=base,
        vllm_model=model,
        vllm_api_key=api_key,
        mix=config.data_gen_mix,
        difficulty=config.data_gen_difficulty,
        require_thinking_trace=config.require_thinking_trace,
        thinking_instruction=config.thinking_instruction,
        artifact_dir=artifact_dir,
    )
    # world_size reserved for future sharding; rank0 materializes full corpus.
    del world_size
    return materialize_data_designer_corpus(out_path, cfg)
