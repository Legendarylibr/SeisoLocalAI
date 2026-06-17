"""GGUF export and Ollama Modelfile generation."""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

DEFAULT_QUANTS = ("q4_k_m", "q8_0", "f16")


def write_ollama_modelfile(
    dest_dir: Path,
    gguf_filename: str,
    *,
    model_name: str = "seiso-model",
    system_prompt: str = "",
    template: str | None = None,
) -> Path:
    """Write Modelfile for `ollama create`."""
    modelfile = dest_dir / "Modelfile"
    lines = [f'FROM ./{gguf_filename}']
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


def convert_hf_dir_to_gguf(source: Path, dest: Path, quant: str, log: Callable[[str], None]) -> bool:
    """Convert a merged HF model directory to GGUF."""
    if dest.exists():
        dest.unlink()

    commands = [
        ["convert_hf_to_gguf", str(source), "--outfile", str(dest), "--outtype", quant],
        ["python3", "-m", "llama_cpp.llama_cpp", "convert", str(source), str(dest)],
    ]
    for cmd in commands:
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=7200)
            if dest.exists() and dest.stat().st_size > 0:
                log(f"GGUF written: {dest}")
                return True
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            continue

    log(f"GGUF conversion failed for {quant} — install llama.cpp convert_hf_to_gguf")
    return False


def export_gguf_from_checkpoint(
    checkpoint: Path,
    output_root: Path,
    quantizations: list[str] | tuple[str, ...],
    *,
    merged_dir: Path | None = None,
    on_log: Callable[[str], None] | None = None,
) -> dict[str, Path]:
    """Merge LoRA checkpoint if needed, convert quantizations, write Ollama Modelfiles."""
    output_root.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        logger.info(msg)
        if on_log:
            on_log(msg)

    if merged_dir is not None and merged_dir.exists():
        results = _export_quants(merged_dir, output_root, quantizations, checkpoint.name, log)
    else:
        with tempfile.TemporaryDirectory(prefix="seiso-gguf-") as tmp:
            merged = Path(tmp) / "merged"
            merged.mkdir()
            from seiso.export.formats import _merge_lora

            _merge_lora(checkpoint, merged, log)
            results = _export_quants(merged, output_root, quantizations, checkpoint.name, log)

    if results:
        first = next(iter(quantizations), "q4_k_m")
        log(f"Ollama: ollama create {checkpoint.name} -f {output_root / first / 'Modelfile'}")
    return results


def _export_quants(
    merged: Path,
    output_root: Path,
    quantizations: list[str] | tuple[str, ...],
    model_name: str,
    log: Callable[[str], None],
) -> dict[str, Path]:
    results: dict[str, Path] = {}
    for quant in quantizations:
        quant_dir = output_root / quant
        quant_dir.mkdir(parents=True, exist_ok=True)
        gguf_path = quant_dir / f"model-{quant}.gguf"
        if convert_hf_dir_to_gguf(merged, gguf_path, quant, log):
            write_ollama_modelfile(quant_dir, gguf_path.name, model_name=model_name)
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
        except Exception as exc:
            log(f"Merge failed: {exc}")
            return []

        paths = _export_quants(merged, output_dir, quantizations, output_dir.name, log)
        return list(paths.values())
