"""Build Ollama import commands from Seiso export outputs."""

from __future__ import annotations

from pathlib import Path


def build_ollama_create_commands(
    outputs: dict[str, str],
    *,
    model_name: str,
) -> list[str]:
    """Return shell commands to import GGUF export artifacts into Ollama."""
    commands: list[str] = []
    safe_name = model_name.strip().replace(" ", "-") or "seiso-model"

    for key, raw_path in sorted(outputs.items()):
        if "gguf" not in key.lower():
            continue
        gguf_path = Path(raw_path)
        modelfile = gguf_path.parent / "Modelfile"
        if not modelfile.is_file():
            continue
        quant = key.removeprefix("gguf_") if key.startswith("gguf_") else key
        name = safe_name if len(outputs) == 1 else f"{safe_name}-{quant}"
        commands.append(f"ollama create {name} -f {modelfile}")

    return commands
