"""GGUF export and Modelfile generation."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_QUANTS = ("q4_k_m", "q8_0", "f16")

# Direct HF→GGUF outtypes supported by the installed convert_hf_to_gguf.py.
DIRECT_CONVERT_OUTTYPES = frozenset(
    {"f32", "f16", "bf16", "q8_0", "tq1_0", "tq2_0", "auto"}
)

# K-quants produced via llama-quantize from an intermediate F16 GGUF.
LLAMA_QUANTIZE_TYPES: dict[str, str] = {
    "q2_k": "Q2_K",
    "q3_k_s": "Q3_K_S",
    "q3_k_m": "Q3_K_M",
    "q3_k_l": "Q3_K_L",
    "q4_0": "Q4_0",
    "q4_k_s": "Q4_K_S",
    "q4_k_m": "Q4_K_M",
    "q5_k_s": "Q5_K_S",
    "q5_k_m": "Q5_K_M",
    "q6_k": "Q6_K",
}

# llama.cpp convert_hf_to_gguf --outtype values supported by Seiso export.
SUPPORTED_GGUF_QUANTS = frozenset(
    {
        "f32",
        "f16",
        "bf16",
        "q8_0",
        "tq1_0",
        "tq2_0",
        "q4_0",
        "q4_1",
        "q5_0",
        "q5_1",
        "q2_k",
        "q3_k_s",
        "q3_k_m",
        "q3_k_l",
        "q4_k_s",
        "q4_k_m",
        "q5_k_s",
        "q5_k_m",
        "q6_k",
        "iq2_xxs",
        "iq2_xs",
        "iq3_xxs",
        "iq1_s",
        "iq4_nl",
        "iq3_s",
        "iq2_s",
        "iq4_xs",
        "iq3_m",
        "q2_k_s",
        "q3_k",
        "q4_k",
        "q5_k",
    }
)

_LABEL_ALIASES = {
    "Q2_K": "q2_k",
    "Q3_K_S": "q3_k_s",
    "Q3_K_M": "q3_k_m",
    "Q3_K_L": "q3_k_l",
    "Q4_K_S": "q4_k_s",
    "Q4_K_M": "q4_k_m",
    "Q5_K_S": "q5_k_s",
    "Q5_K_M": "q5_k_m",
    "Q6_K": "q6_k",
    "Q8_0": "q8_0",
    "F16": "f16",
    "F32": "f32",
    "BF16": "bf16",
}


def normalize_gguf_quant(label: str) -> str:
    """Normalize a GGUF quant label to llama.cpp --outtype form."""
    raw = label.strip()
    upper = raw.upper().replace("-", "_")
    if upper in _LABEL_ALIASES:
        return _LABEL_ALIASES[upper]
    lowered = raw.lower().replace("-", "_")
    return lowered


def normalize_gguf_quants(quantizations: list[str] | tuple[str, ...]) -> list[str]:
    """Normalize and deduplicate GGUF quant labels, falling back when unknown."""
    seen: set[str] = set()
    out: list[str] = []
    for label in quantizations:
        quant = normalize_gguf_quant(label)
        if quant not in SUPPORTED_GGUF_QUANTS:
            logger.warning(
                "Unknown GGUF quant %r — using as-is (may fail at convert time)", label
            )
        if quant not in seen:
            seen.add(quant)
            out.append(quant)
    return out or ["q4_k_m"]


def resolve_llama_quantize_binary() -> Path | None:
    llama_cpp_dir = os.environ.get("LLAMA_CPP_DIR", "").strip()
    if llama_cpp_dir:
        candidate = Path(llama_cpp_dir) / "build" / "bin" / "llama-quantize"
        if candidate.is_file():
            return candidate
    if path := shutil.which("llama-quantize"):
        return Path(path)
    return None


def convert_outtype_for_hf_export(quant: str) -> str | None:
    """Return a direct convert_hf_to_gguf outtype, or None when quantize-from-f16 is required."""
    normalized = normalize_gguf_quant(quant)
    if normalized in DIRECT_CONVERT_OUTTYPES:
        return normalized
    if normalized in LLAMA_QUANTIZE_TYPES:
        return None
    return normalized if normalized in DIRECT_CONVERT_OUTTYPES else None


def quantize_gguf_file(
    source: Path,
    dest: Path,
    quant: str,
    log: Callable[[str], None],
) -> bool:
    """Re-quantize an existing GGUF (typically F16) with llama-quantize."""
    normalized = normalize_gguf_quant(quant)
    quant_type = LLAMA_QUANTIZE_TYPES.get(normalized)
    binary = resolve_llama_quantize_binary()
    if quant_type is None or binary is None:
        return False
    if dest.exists():
        dest.unlink()
    cmd = [str(binary), str(source), str(dest), quant_type]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=7200)
    except (
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        FileNotFoundError,
    ) as exc:
        detail = getattr(exc, "stderr", "") or getattr(exc, "stdout", "") or str(exc)
        log(f"llama-quantize failed for {quant}: {str(detail)[:300]}")
        return False
    if dest.is_file() and dest.stat().st_size > 0:
        log(f"GGUF written: {dest} ({quant_type})")
        return True
    return False


def resolve_gguf_converter() -> list[list[str]]:
    """Return candidate llama.cpp HF→GGUF converter command prefixes (most preferred first)."""
    candidates: list[list[str]] = []

    llama_cpp_dir = os.environ.get("LLAMA_CPP_DIR", "").strip()
    if llama_cpp_dir:
        script = Path(llama_cpp_dir) / "convert_hf_to_gguf.py"
        if script.is_file():
            py = shutil.which("python3") or shutil.which("python") or "python3"
            candidates.append([py, str(script)])

    for name in ("convert_hf_to_gguf", "convert-hf-to-gguf"):
        if path := shutil.which(name):
            candidates.append([path])

    return candidates


def write_modelfile(
    dest_dir: Path,
    gguf_filename: str,
    *,
    model_name: str = "seiso-model",
    system_prompt: str = "",
    template: str | None = None,
) -> Path:
    """Write a Modelfile for use with llama.cpp / Modelfile-aware tooling."""
    modelfile = dest_dir / "Modelfile"
    lines = [f"FROM ./{gguf_filename}"]
    if template:
        lines.append(f'TEMPLATE """{template}"""')
    if system_prompt:
        lines.append(f'SYSTEM """{system_prompt}"""')
    lines.extend(
        [
            "PARAMETER temperature 0.7",
            "PARAMETER top_p 0.9",
            "PARAMETER top_k 40",
            "PARAMETER stop <|eot_id|>",
            "PARAMETER stop </s>",
        ]
    )
    modelfile.write_text("\n".join(lines) + "\n")
    return modelfile


def convert_hf_dir_to_gguf(
    source: Path, dest: Path, quant: str, log: Callable[[str], None]
) -> bool:
    """Convert a merged HF model directory to GGUF."""
    quant = normalize_gguf_quant(quant)
    if dest.exists():
        dest.unlink()

    direct = convert_outtype_for_hf_export(quant)
    if direct is not None:
        return _convert_hf_dir_direct(source, dest, direct, log)

    if quant in LLAMA_QUANTIZE_TYPES:
        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as tmp:
            f16_path = Path(tmp.name)
        try:
            if not _convert_hf_dir_direct(source, f16_path, "f16", log):
                return False
            return quantize_gguf_file(f16_path, dest, quant, log)
        finally:
            f16_path.unlink(missing_ok=True)

    log(
        f"GGUF conversion failed for {quant}: unsupported quant for installed llama.cpp"
    )
    return False


def _convert_hf_dir_direct(
    source: Path, dest: Path, outtype: str, log: Callable[[str], None]
) -> bool:
    converters = resolve_gguf_converter()
    if not converters:
        log(
            "GGUF conversion unavailable — install llama.cpp and add convert_hf_to_gguf to PATH, "
            "or set LLAMA_CPP_DIR to a llama.cpp checkout"
        )
        return False

    errors: list[str] = []
    for prefix in converters:
        cmd = [*prefix, str(source), "--outfile", str(dest), "--outtype", outtype]
        try:
            subprocess.run(
                cmd, check=True, capture_output=True, text=True, timeout=7200
            )
            if dest.exists() and dest.stat().st_size > 0:
                log(f"GGUF written: {dest} ({outtype})")
                return True
            errors.append(f"{cmd[0]}: output file missing or empty")
        except FileNotFoundError:
            errors.append(f"{cmd[0]}: not found")
        except subprocess.TimeoutExpired:
            errors.append(f"{cmd[0]}: timed out")
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or str(exc)).strip()
            errors.append(f"{cmd[0]}: {detail[:300]}")

    log(f"GGUF conversion failed for {outtype}: {'; '.join(errors[:3])}")
    return False


def export_gguf_from_checkpoint(
    checkpoint: Path,
    output_root: Path,
    quantizations: list[str] | tuple[str, ...],
    *,
    merged_dir: Path | None = None,
    on_log: Callable[[str], None] | None = None,
) -> dict[str, Path]:
    """Merge LoRA checkpoint if needed, convert quantizations, write Modelfiles."""
    output_root.mkdir(parents=True, exist_ok=True)
    quants = normalize_gguf_quants(quantizations)

    def log(msg: str) -> None:
        logger.info(msg)
        if on_log:
            on_log(msg)

    if merged_dir is not None and merged_dir.exists():
        results = _export_quants(merged_dir, output_root, quants, checkpoint.name, log)
    else:
        with tempfile.TemporaryDirectory(prefix="seiso-gguf-") as tmp:
            merged = Path(tmp) / "merged"
            merged.mkdir()
            from seiso.export.formats import merge_lora_checkpoint

            merge_lora_checkpoint(checkpoint, merged, log)
            results = _export_quants(merged, output_root, quants, checkpoint.name, log)

    if not results:
        raise RuntimeError(
            f"GGUF export produced no artifacts for quants: {', '.join(quants)}. "
            "Install llama.cpp convert_hf_to_gguf or set LLAMA_CPP_DIR."
        )

    first = next(iter(quants), "q4_k_m")
    log(f"Modelfile written: {output_root / first / 'Modelfile'}")
    return results


def _export_quants(
    merged: Path,
    output_root: Path,
    quantizations: list[str],
    model_name: str,
    log: Callable[[str], None],
) -> dict[str, Path]:
    results: dict[str, Path] = {}
    for quant in quantizations:
        quant_dir = output_root / quant
        quant_dir.mkdir(parents=True, exist_ok=True)
        gguf_path = quant_dir / f"model-{quant}.gguf"
        if convert_hf_dir_to_gguf(merged, gguf_path, quant, log):
            write_modelfile(quant_dir, gguf_path.name, model_name=model_name)
            results[f"gguf_{quant}"] = gguf_path
    return results


def export_gguf(
    model: Any,
    tokenizer: Any,
    output_dir: Path,
    quantizations: list[str] | tuple[str, ...] = DEFAULT_QUANTS,
    *,
    on_log: Callable[[str], None] | None = None,
) -> list[Path]:
    """Export in-memory model to GGUF quantizations."""
    output_dir.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        logger.info(msg)
        if on_log:
            on_log(msg)

    with tempfile.TemporaryDirectory(prefix="seiso-gguf-") as tmp:
        merged = Path(tmp) / "merged"
        merged.mkdir()
        try:
            from peft import PeftModel

            if isinstance(model, PeftModel):
                log("Merging LoRA for GGUF export...")
                merged_model = model.merge_and_unload()
                merged_model.save_pretrained(str(merged))
            else:
                model.save_pretrained(str(merged))
            tokenizer.save_pretrained(str(merged))
            from seiso.memory.protection import release_cached_memory

            release_cached_memory()
        except Exception as exc:
            log(f"Merge failed: {exc}")
            return []

        paths = _export_quants(
            merged,
            output_dir,
            normalize_gguf_quants(quantizations),
            output_dir.name,
            log,
        )
        return list(paths.values())
