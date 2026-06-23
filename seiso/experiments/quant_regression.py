"""Train multiple QLoRA quants of one model, then run RL quant to measure regression."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any

from seiso.export.formats import ExportFormat, ExportOptions, export_checkpoint
from seiso.export.gguf import normalize_gguf_quants, resolve_gguf_converter
from seiso.training.config import QuantMode, TrainConfig, run_training

DEFAULT_TRAIN_QUANTS: tuple[str, ...] = ("4bit", "8bit", "16bit")
DEFAULT_GGUF_QUANTS: tuple[str, ...] = ("q4_k_m", "q8_0", "f16")
DEFAULT_ROUTE_HARDWARE: tuple[str, ...] = ("gpu",)
LLAMA_CLI_PYTHON_SHIM = Path(__file__).resolve().parents[2] / "scripts" / "llama_cli_python_shim.py"

# Seiso export folder → adaptive_quant RouteCatalog quant labels.
ROUTE_QUANT_LABELS: dict[str, str] = {
    "q4_0": "Q4_0",
    "q4_k_m": "Q4_K_M",
    "q8_0": "Q8_0",
    "f16": "F16",
    "bf16": "BF16",
}

# Map Seiso GGUF folder names to router @qN suffixes (adaptive_quant routing).
GGUF_ROUTE_BITS: dict[str, int] = {
    "q2_k": 2,
    "q3_k_s": 3,
    "q3_k_m": 3,
    "q3_k_l": 3,
    "q4_0": 4,
    "q4_k_s": 4,
    "q4_k_m": 4,
    "q5_0": 5,
    "q5_k_s": 5,
    "q5_k_m": 5,
    "q6_k": 6,
    "q8_0": 8,
    "f16": 16,
    "bf16": 16,
}


@dataclass
class QuantRegressionRow:
    train_quant: str
    checkpoint: str
    train_loss: float | None = None
    eval_loss: float | None = None
    export_dir: str | None = None
    rl_job_id: str | None = None
    rl_output_dir: str | None = None
    route_eval_path: str | None = None
    backend: str = "llama_cpp"
    eval_mean_reward: float | None = None
    eval_mean_perplexity: float | None = None
    recommended_route: str | None = None
    recommended_quant: str | None = None
    reward_regression: float | None = None
    perplexity_regression: float | None = None
    mean_selected_memory_mb: float | None = None
    router_routes: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class QuantRegressionReport:
    model_id: str
    study_dir: str
    rows: list[QuantRegressionRow] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "study_dir": self.study_dir,
            "rows": [asdict(row) for row in self.rows],
        }


def gguf_route_bits(quant_label: str) -> int:
    normalized = normalize_gguf_quants([quant_label])[0]
    bits = GGUF_ROUTE_BITS.get(normalized)
    if bits is None:
        raise ValueError(f"No router bit mapping for GGUF quant {quant_label!r}")
    return bits


def build_llama_cpp_router_routes(gguf_paths: dict[str, Path]) -> list[str]:
    """Build ``llama_cpp:/path/model.gguf@qN`` routes from export paths."""
    routes: list[str] = []
    for quant_label, path in sorted(gguf_paths.items()):
        bits = gguf_route_bits(quant_label)
        routes.append(f"llama_cpp:{path.resolve()}@q{bits}")
    return routes


def _resolve_regression_bounds(
    base_config: TrainConfig,
    *,
    max_reward_regression: float | None,
    max_perplexity_regression: float | None,
) -> tuple[float, float]:
    extra = base_config.extra if isinstance(base_config.extra, dict) else {}
    reward = (
        float(max_reward_regression)
        if max_reward_regression is not None
        else float(extra.get("max_reward_regression", 0.10))
    )
    perplexity = (
        float(max_perplexity_regression)
        if max_perplexity_regression is not None
        else float(extra.get("max_perplexity_regression", 0.05))
    )
    return reward, perplexity


def _resolve_train_config(train_out: Path, base_config: TrainConfig) -> TrainConfig:
    snapshot = train_out / "train_config_snapshot.json"
    if snapshot.is_file():
        return TrainConfig.model_validate(json.loads(snapshot.read_text(encoding="utf-8")))
    return base_config


def build_eval_route_prompt_library(
    train_out: Path,
    base_config: TrainConfig,
    *,
    max_prompts: int = 16,
) -> list[Any]:
    """Build llama.cpp route prompts from the training eval split (e.g. MetaMathQA holdout)."""
    from adaptive_quant.prompts import default_prompt_library
    from adaptive_quant.types import PromptSample
    from seiso.models.chat_format import extract_messages, format_messages_for_prompt
    from seiso.training.config import DatasetFormat
    from seiso.training.datasets import detect_format, load_training_dataset
    from transformers import AutoTokenizer

    cfg = _resolve_train_config(train_out, base_config)
    raw = load_training_dataset(cfg.dataset, sandbox_root=cfg.sandbox_root)
    max_total = cfg.extra.get("max_samples") if isinstance(cfg.extra, dict) else None
    if isinstance(max_total, int) and max_total > 0 and len(raw) > max_total:
        raw = raw.select(range(max_total))
    if cfg.eval_split_ratio > 0 and len(raw) > 10:
        eval_ds = raw.train_test_split(test_size=cfg.eval_split_ratio, seed=cfg.seed)["test"]
    else:
        eval_ds = raw

    ds_fmt = cfg.dataset_format
    if ds_fmt == DatasetFormat.AUTO and len(eval_ds) > 0:
        ds_fmt = detect_format(eval_ds[0])

    tokenizer = AutoTokenizer.from_pretrained(str(cfg.model_id), trust_remote_code=False)
    limit = max(1, max_prompts)
    prompts: list[PromptSample] = []
    for index, sample in enumerate(eval_ds):
        if len(prompts) >= limit:
            break
        messages = extract_messages(sample, ds_fmt)
        if not messages:
            continue
        if messages[-1].get("role") == "assistant":
            prompt_messages = messages[:-1]
        else:
            prompt_messages = messages
        prompt_text = format_messages_for_prompt(
            prompt_messages,
            tokenizer,
            add_generation_prompt=True,
        ).strip()
        if not prompt_text:
            continue
        prompts.append(PromptSample(f"eval_{index:03d}", prompt_text, "math"))

    if prompts:
        return prompts
    return default_prompt_library()[:limit]


def _first_gguf_path(gguf_paths: dict[str, Path]) -> Path | None:
    for key in ("q4_k_m", "q8_0", "f16"):
        path = gguf_paths.get(key) or gguf_paths.get(f"gguf_{key}")
        if path and Path(path).is_file():
            return Path(path)
    for path in gguf_paths.values():
        if Path(path).is_file():
            return Path(path)
    return None


def export_merged_checkpoint(
    checkpoint: Path,
    export_root: Path,
    *,
    on_log: Callable[[str], None] | None = None,
) -> Path:
    """Merge LoRA adapter to HF weights for downstream eval/export."""
    from seiso.export.formats import merge_lora_checkpoint

    export_root.mkdir(parents=True, exist_ok=True)
    merged = export_root / "merged"
    merge_lora_checkpoint(checkpoint, merged, on_log or (lambda _m: None))
    return merged


def export_checkpoint_ggufs(
    checkpoint: Path,
    export_root: Path,
    *,
    gguf_quants: list[str] | tuple[str, ...] = DEFAULT_GGUF_QUANTS,
    on_log: Callable[[str], None] | None = None,
) -> dict[str, Path]:
    """Merge LoRA checkpoint and export GGUF variants when llama.cpp is available."""
    if not resolve_gguf_converter():
        if on_log:
            on_log("GGUF converter unavailable — skipping GGUF export (set LLAMA_CPP_DIR)")
        return {}

    export_root.mkdir(parents=True, exist_ok=True)
    results = export_checkpoint(
        ExportOptions(
            checkpoint=checkpoint,
            output_dir=export_root,
            formats=[ExportFormat.MERGED, ExportFormat.GGUF],
            gguf_quantizations=list(gguf_quants),
            sandbox_root=checkpoint.parent.parent,
        ),
        on_log=on_log,
    )
    gguf_paths: dict[str, Path] = {}
    for key, value in results.items():
        if key.startswith("gguf_") and Path(value).is_file():
            gguf_paths[key.removeprefix("gguf_")] = Path(value)
        elif key == ExportFormat.GGUF.value:
            continue
    return gguf_paths


def resolve_llama_cpp_python_shim() -> Path | None:
    shim = LLAMA_CLI_PYTHON_SHIM
    return shim if shim.is_file() else None


def llama_cpp_python_gpu_ready() -> bool:
    try:
        from seiso.inference.llamacpp_install import llamacpp_gpu_offload_supported, llamacpp_import_ok

        ok, _err = llamacpp_import_ok()
        return ok and llamacpp_gpu_offload_supported()
    except ImportError:
        return False


def _llama_cli_binary_has_gpu(binary: Path) -> bool:
    try:
        proc = __import__("subprocess").run(
            [str(binary), "--version"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        combined = f"{proc.stdout}\n{proc.stderr}"
        if "compiled without GPU support" in combined.lower():
            return False
        if "cuda : on" in combined.lower() or "cublas" in combined.lower():
            return True
        return "cuda" in combined.lower() and "off" not in combined.lower()
    except OSError:
        return False


def resolve_llama_cpp_binary(explicit: str | None = None) -> Path | None:
    if explicit:
        path = Path(explicit).expanduser()
        if path.is_file():
            if path.name == "llama_cli_python_shim.py" or _llama_cli_binary_has_gpu(path):
                return path
        shim = resolve_llama_cpp_python_shim()
        if shim is not None and llama_cpp_python_gpu_ready():
            return shim
        return path if path.is_file() else None

    candidates: list[Path] = []
    env = os.environ.get("LLAMA_CPP_BINARY", "").strip()
    if env:
        candidates.append(Path(env))
    llama_cpp_dir = os.environ.get("LLAMA_CPP_DIR", "").strip()
    if llama_cpp_dir:
        candidates.append(Path(llama_cpp_dir) / "build" / "bin" / "llama-cli")
    if path := shutil.which("llama-cli"):
        candidates.append(Path(path))

    for candidate in candidates:
        if candidate.is_file() and _llama_cli_binary_has_gpu(candidate):
            return candidate

    if llama_cpp_python_gpu_ready():
        shim = resolve_llama_cpp_python_shim()
        if shim is not None:
            return shim

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def resolve_study_backend(requested: str | None) -> str:
    if requested:
        return requested
    return "hf"


def llama_cpp_ready() -> bool:
    if not resolve_gguf_converter():
        return False
    binary = resolve_llama_cpp_binary()
    if binary is None:
        return False
    if binary.name == "llama_cli_python_shim.py":
        return llama_cpp_python_gpu_ready()
    return _llama_cli_binary_has_gpu(binary)


def build_route_catalog(
    gguf_paths: dict[str, Path],
    *,
    route_repo_id: str = "local/quant-regression-study",
) -> Any:
    from adaptive_quant.model_routes import ModelRoute, RouteCatalog

    routes: list[Any] = []
    for quant_key, path in sorted(gguf_paths.items()):
        normalized = normalize_gguf_quants([quant_key])[0]
        quant_label = ROUTE_QUANT_LABELS.get(normalized)
        if quant_label is None:
            raise ValueError(f"No RouteCatalog quant label for GGUF folder {quant_key!r}")
        if not path.is_file():
            raise FileNotFoundError(f"GGUF route file missing: {path}")
        routes.append(
            ModelRoute(
                route_id=f"gguf_{normalized}",
                repo_id=route_repo_id,
                quant_label=quant_label,
                filename=path.name,
                local_path=str(path.resolve()),
                size_mb=max(1.0, path.stat().st_size / (1024 * 1024)),
                hardware_hints=DEFAULT_ROUTE_HARDWARE,
            )
        )
    if len(routes) < 2:
        raise ValueError("Route regression requires at least two exported GGUF variants")
    return RouteCatalog(routes=routes)


def summarize_route_report(report: dict[str, Any]) -> dict[str, Any]:
    rows = report.get("rows") if isinstance(report.get("rows"), list) else []
    recommendations = (
        report.get("recommendations") if isinstance(report.get("recommendations"), list) else []
    )
    selected = [rec for rec in recommendations if isinstance(rec, dict) and rec.get("route_id")]

    rewards = [_finite(row.get("reward")) for row in rows if isinstance(row, dict)]
    perplexities = [_finite(row.get("perplexity")) for row in rows if isinstance(row, dict)]
    reward_regs = [_finite(rec.get("reward_regression")) for rec in selected]
    ppl_regs = [_finite(rec.get("perplexity_regression")) for rec in selected]
    reward_regs = [v for v in reward_regs if v is not None]
    ppl_regs = [v for v in ppl_regs if v is not None]

    best_rec = selected[0] if selected else {}
    return {
        "eval_mean_reward": mean([v for v in rewards if v is not None]) if rewards else None,
        "eval_mean_perplexity": mean([v for v in perplexities if v is not None])
        if perplexities
        else None,
        "recommended_route": best_rec.get("route_id"),
        "recommended_quant": best_rec.get("quant_label"),
        "reward_regression": mean(reward_regs) if reward_regs else _finite(best_rec.get("reward_regression")),
        "perplexity_regression": mean(ppl_regs) if ppl_regs else _finite(best_rec.get("perplexity_regression")),
        "mean_selected_memory_mb": _finite(report.get("mean_selected_memory_mb")),
    }


def run_route_regression_eval(
    *,
    data_dir: Path,
    gguf_paths: dict[str, Path],
    llama_cpp_binary: str | Path,
    router_routes: list[str],
    train_out: Path | None = None,
    base_config: TrainConfig | None = None,
    on_log: Callable[[str], None] | None = None,
    llama_cpp_timeout_s: float = 600.0,
    max_reward_regression: float = 0.10,
    max_perplexity_regression: float = 0.05,
    route_prompt_limit: int = 16,
    hardware: tuple[str, ...] = DEFAULT_ROUTE_HARDWARE,
    route_repo_id: str = "local/quant-regression-study",
    rl_user_id: str = "quant_regression",
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Evaluate GGUF routes with real llama.cpp and bounded regression selection."""
    from seiso.rl_quant.bootstrap import ensure_adaptive_quant_importable
    from seiso.rl_quant.config_builder import build_framework_config

    ensure_adaptive_quant_importable()
    from adaptive_quant.prompts import PromptLibrary
    from adaptive_quant.route_pipeline import evaluate_routes_for_prompts, validate_local_route_models
    from adaptive_quant.types import HardwareType

    catalog = build_route_catalog(gguf_paths, route_repo_id=route_repo_id)
    validate_local_route_models(catalog)

    job_id = str(uuid.uuid4())[:12]
    primary = _first_gguf_path(gguf_paths)
    payload: dict[str, Any] = {
        "preset": "minimal",
        "run_name": f"route_regression_{job_id}",
        "backend": "llama_cpp",
        "llama_cpp_binary": str(llama_cpp_binary),
        "llama_cpp_timeout_s": float(llama_cpp_timeout_s),
        "router_enabled": True,
        "router_routes": router_routes,
        "hardware_modes": hardware,
        "route_hf_allowed_repos": [route_repo_id],
        "auto_sweep": False,
        "write_research_report": False,
    }
    if primary:
        payload["gguf_path"] = str(primary)

    config = build_framework_config(
        job_id=job_id,
        user_id=rl_user_id,
        data_dir=data_dir,
        payload=payload,
    )

    if train_out is not None and base_config is not None:
        prompts = build_eval_route_prompt_library(
            train_out,
            base_config,
            max_prompts=max(1, route_prompt_limit),
        )
        prompt_source = "eval_split"
    else:
        from adaptive_quant.prompts import default_prompt_library

        prompts = default_prompt_library()[: max(1, route_prompt_limit)]
        prompt_source = "default_library"
    library = PromptLibrary(prompts)
    hw = tuple(HardwareType(value) for value in hardware)

    if on_log:
        on_log(
            f"Route regression ({prompt_source}): {len(catalog.routes)} GGUF routes × "
            f"{len(prompts)} prompts × {len(hw)} hardware "
            f"(reward_reg≤{max_reward_regression:.2f}, ppl_reg≤{max_perplexity_regression:.2f})"
        )

    report = evaluate_routes_for_prompts(
        config,
        catalog=catalog,
        prompt_library=library,
        hardware=hw,
        max_reward_regression=max_reward_regression,
        max_perplexity_regression=max_perplexity_regression,
    )

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "route_regression.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        report["artifact_path"] = str(report_path)

    report["output_dir"] = str(
        output_dir
        or (data_dir / "rl_quant" / "quant_regression" / job_id)
    )
    report["job_id"] = job_id
    return report


def _read_training_metrics(checkpoint: Path) -> tuple[float | None, float | None]:
    metrics = checkpoint.parent / "metrics.jsonl"
    if not metrics.is_file():
        return None, None
    train_loss: float | None = None
    eval_loss: float | None = None
    for line in metrics.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("type") == "eval" or row.get("eval_loss") is not None:
            eval_loss = _finite(row.get("eval_loss") or row.get("loss")) or eval_loss
        train_loss = _finite(row.get("train_loss") or row.get("loss")) or train_loss
    return train_loss, eval_loss


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out or out in (float("inf"), float("-inf")):  # NaN / inf
        return None
    return out


def run_quant_regression_study(
    base_config: TrainConfig,
    *,
    data_dir: Path,
    study_dir: Path | None = None,
    train_quants: list[str] | tuple[str, ...] = DEFAULT_TRAIN_QUANTS,
    gguf_quants: list[str] | tuple[str, ...] = DEFAULT_GGUF_QUANTS,
    deploy_quants: list[str] | tuple[str, ...] = ("4bit", "8bit", "16bit"),
    measurement: str | None = None,
    rl_backend: str | None = None,
    llama_cpp_binary: str | None = None,
    llama_cpp_timeout_s: float = 600.0,
    max_reward_regression: float | None = None,
    max_perplexity_regression: float | None = None,
    route_prompt_limit: int = 16,
    max_eval_samples: int = 64,
    skip_training: bool = False,
    skip_rl: bool = False,
    on_log: Callable[[str], None] | None = None,
) -> QuantRegressionReport:
    """Train one model at several quants, then measure deploy-quant regression."""
    from seiso.experiments.hf_deploy_regression import (
        run_hf_deploy_quant_regression,
        summarize_hf_deploy_report,
    )
    from seiso.memory.protection import apply_training_memory_guards

    mode = rl_backend or measurement or resolve_study_backend(None)
    if mode in {"llama_cpp", "both"} and not llama_cpp_ready():
        if mode == "llama_cpp":
            raise RuntimeError(
                "llama_cpp measurement requires CUDA-enabled llama-cli (or llama-cpp-python GPU shim) "
                "and convert_hf_to_gguf. Use --measurement hf (default) for GPU eval on merged weights."
            )
        log("llama_cpp backend unavailable — running HF measurement only")
        mode = "hf"
    llama_bin = resolve_llama_cpp_binary(llama_cpp_binary)

    root = (study_dir or base_config.output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    reward_bound, perplexity_bound = _resolve_regression_bounds(
        base_config,
        max_reward_regression=max_reward_regression,
        max_perplexity_regression=max_perplexity_regression,
    )
    extra = base_config.extra if isinstance(base_config.extra, dict) else {}
    route_repo_id = str(extra.get("route_repo_id") or "local/quant-regression-study")
    rl_user_id = str(extra.get("rl_user_id") or "quant_regression")

    report = QuantRegressionReport(model_id=base_config.model_id, study_dir=str(root))
    manifest_path = root / "quant_regression_manifest.json"
    existing: dict[str, Any] = {}
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))

    def log(msg: str) -> None:
        if on_log:
            on_log(msg)

    for quant_label in train_quants:
        quant = QuantMode(quant_label)
        row = QuantRegressionRow(train_quant=quant_label, checkpoint="", backend=mode)
        train_out = root / f"train-{quant_label}"
        checkpoint = _resolve_checkpoint(train_out)

        try:
            if skip_training and checkpoint is None:
                raise FileNotFoundError(f"No checkpoint under {train_out}")

            if not skip_training:
                cfg = apply_training_memory_guards(
                    base_config.model_copy(
                        update={"quant": quant, "output_dir": train_out},
                        deep=True,
                    )
                )
                log(f"Training {cfg.model_id} quant={quant_label} → {train_out}")
                checkpoint = run_training(cfg, on_log=on_log)
                row.checkpoint = str(checkpoint)
            else:
                row.checkpoint = str(checkpoint)

            row.train_loss, row.eval_loss = _read_training_metrics(Path(row.checkpoint))

            export_dir = train_out / "export"
            merged_dir = export_merged_checkpoint(Path(row.checkpoint), export_dir, on_log=log)
            row.export_dir = str(export_dir)

            if skip_rl:
                report.rows.append(row)
                _persist_manifest(manifest_path, report, existing)
                continue

            if mode in {"hf", "both"}:
                log(
                    f"HF deploy-quant regression quant={quant_label} "
                    f"deploy={list(deploy_quants)} samples={max_eval_samples}"
                )
                deploy_report = run_hf_deploy_quant_regression(
                    merged_dir,
                    train_out=train_out,
                    base_config=base_config,
                    deploy_quants=deploy_quants,
                    max_eval_samples=max_eval_samples,
                    max_reward_regression=reward_bound,
                    max_perplexity_regression=perplexity_bound,
                    on_log=on_log,
                    output_dir=export_dir,
                )
                row.route_eval_path = deploy_report.get("artifact_path")
                metrics = summarize_hf_deploy_report(deploy_report)
                row.backend = "both" if mode == "both" else "hf"

            if mode in {"llama_cpp", "both"}:
                from seiso.export.gguf import export_gguf_from_checkpoint

                gguf_results = export_gguf_from_checkpoint(
                    Path(row.checkpoint),
                    export_dir,
                    gguf_quants,
                    merged_dir=merged_dir,
                    on_log=on_log,
                )
                gguf_paths = {
                    key.removeprefix("gguf_"): Path(path)
                    for key, path in gguf_results.items()
                    if key.startswith("gguf_")
                }
                if len(gguf_paths) < 2:
                    raise RuntimeError(
                        f"Need ≥2 GGUF files for route regression; got {list(gguf_paths)}"
                    )
                routes = build_llama_cpp_router_routes(gguf_paths)
                row.router_routes = routes
                row.rl_job_id = str(uuid.uuid4())[:12]
                log(
                    f"Route regression quant={quant_label} routes={len(routes)} "
                    f"timeout={llama_cpp_timeout_s:.0f}s"
                )
                route_report = run_route_regression_eval(
                    data_dir=data_dir,
                    gguf_paths=gguf_paths,
                    llama_cpp_binary=llama_bin or "",
                    router_routes=routes,
                    train_out=train_out,
                    base_config=base_config,
                    on_log=on_log,
                    llama_cpp_timeout_s=llama_cpp_timeout_s,
                    max_reward_regression=reward_bound,
                    max_perplexity_regression=perplexity_bound,
                    route_prompt_limit=route_prompt_limit,
                    route_repo_id=route_repo_id,
                    rl_user_id=rl_user_id,
                    output_dir=export_dir,
                )
                llama_path = route_report.get("artifact_path")
                if mode == "both":
                    row.rl_output_dir = str(route_report.get("output_dir") or "")
                    if llama_path and Path(llama_path).is_file():
                        alt = export_dir / "route_regression_llama.json"
                        shutil.copy2(llama_path, alt)
                        row.rl_output_dir = str(alt)
                else:
                    row.route_eval_path = route_report.get("artifact_path")
                    row.rl_output_dir = str(route_report.get("output_dir") or "")
                    metrics = summarize_route_report(route_report)
                    row.backend = "llama_cpp"

            row.eval_mean_reward = metrics.get("eval_mean_reward")
            row.eval_mean_perplexity = metrics.get("eval_mean_perplexity")
            row.recommended_route = metrics.get("recommended_route")
            row.recommended_quant = metrics.get("recommended_quant")
            row.reward_regression = metrics.get("reward_regression")
            row.perplexity_regression = metrics.get("perplexity_regression")
            row.mean_selected_memory_mb = metrics.get("mean_selected_memory_mb")
        except Exception as exc:
            row.error = str(exc)
            log(f"quant={quant_label} failed: {exc}")

        report.rows.append(row)
        _persist_manifest(manifest_path, report, existing)

    report_path = root / "quant_regression_report.json"
    report_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return report


def _resolve_checkpoint(train_out: Path) -> Path | None:
    if not train_out.exists():
        return None
    manifest = train_out / "seiso_manifest.json"
    if manifest.is_file():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        ckpt = data.get("checkpoint_path") or data.get("output_dir")
        if ckpt and Path(ckpt).exists():
            return Path(ckpt)
    candidates = sorted(train_out.glob("checkpoint-*"), reverse=True)
    for candidate in candidates:
        if (candidate / "adapter_config.json").is_file() or (candidate / "config.json").is_file():
            return candidate
    if (train_out / "adapter_config.json").is_file():
        return train_out
    return None


def _persist_manifest(path: Path, report: QuantRegressionReport, existing: dict[str, Any]) -> None:
    payload = {**existing, **report.to_dict()}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def format_report_table(report: QuantRegressionReport) -> str:
    """Render a compact ASCII table for terminal output."""
    headers = (
        "train_quant",
        "train_loss",
        "backend",
        "rec_quant",
        "eval_reward",
        "eval_ppl",
        "reward_reg",
        "ppl_reg",
        "mem_mb",
    )
    lines = [" | ".join(headers), "-" * 96]
    for row in report.rows:
        if row.error:
            lines.append(f"{row.train_quant} | ERROR: {row.error}")
            continue
        lines.append(
            " | ".join(
                [
                    row.train_quant,
                    _fmt(row.train_loss),
                    row.backend,
                    (row.recommended_quant or "-")[:8],
                    _fmt(row.eval_mean_reward),
                    _fmt(row.eval_mean_perplexity),
                    _fmt(row.reward_regression),
                    _fmt(row.perplexity_regression),
                    _fmt(row.mean_selected_memory_mb),
                ]
            )
        )
    return "\n".join(lines)


def _fmt(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}"
