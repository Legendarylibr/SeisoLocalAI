"""Hugging Face model card and metadata generation for Seiso exports."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from seiso.io.jsonl import read_json_file

_FINETUNE_TYPES = frozenset(
    {"lora", "qlora", "full", "embedding", "slime", "nemo_rl", "compress"}
)


@dataclass
class HubModelMetadata:
    """Fields used to build a Hugging Face repo id and model card."""

    username: str
    model_name: str
    author: str
    license: str = "apache-2.0"
    base_model: str | None = None
    description: str = ""
    tags: list[str] = field(default_factory=list)
    quantizations: list[str] = field(default_factory=list)
    library_name: str = "transformers"
    pipeline_tag: str = "text-generation"
    finetune_type: str | None = None
    export_formats: list[str] = field(default_factory=list)
    seiso_job_id: str | None = None
    seiso_source: str | None = None

    @property
    def repo_id(self) -> str:
        return f"{self.username.strip()}/{self.model_name.strip()}"

    def validate(self) -> None:
        if not self.username.strip():
            raise ValueError("Hugging Face username is required")
        if not self.model_name.strip():
            raise ValueError("Model name is required")
        if not self.author.strip():
            raise ValueError("Author is required")
        if "/" in self.model_name:
            raise ValueError("Model name must not contain '/'")
        if not re.match(r"^[A-Za-z0-9][A-Za-z0-9._-]*$", self.model_name.strip()):
            raise ValueError(
                "Model name must start with alphanumeric and contain only letters, digits, '.', '_', or '-'"
            )
        if self.finetune_type and self.finetune_type not in _FINETUNE_TYPES:
            raise ValueError(
                f"finetune_type must be one of: {', '.join(sorted(_FINETUNE_TYPES))}"
            )

    def to_card_dict(self) -> dict[str, Any]:
        self.validate()
        tags = list(dict.fromkeys([*self.tags, "seiso"]))
        if self.quantizations:
            tags.append("gguf")
        if self.finetune_type:
            tags.append(self.finetune_type)
        else:
            tags.append("finetuned")

        card: dict[str, Any] = {
            "license": self.license,
            "tags": tags,
            "library_name": self.library_name,
            "pipeline_tag": self.pipeline_tag,
        }
        if self.base_model:
            card["base_model"] = self.base_model
        if self.finetune_type in {"lora", "qlora"}:
            card["base_model"] = card.get("base_model") or self.base_model
        return card


def _yaml_frontmatter(card: dict[str, Any]) -> str:
    lines = ["---"]
    for key, value in card.items():
        if isinstance(value, list):
            lines.append(f"{key}:")
            for item in value:
                lines.append(f"  - {item}")
        else:
            lines.append(f"{key}: {value}")
    lines.append("---")
    return "\n".join(lines)


def render_readme(
    meta: HubModelMetadata, *, extra: dict[str, Any] | None = None
) -> str:
    """Render README.md body with YAML frontmatter for Hugging Face."""
    meta.validate()
    card = meta.to_card_dict()
    title = meta.model_name.replace("-", " ").replace("_", " ").title()
    quant_line = ""
    if meta.quantizations:
        quant_line = (
            "\n## Quantizations\n\n"
            + "\n".join(f"- `{q}`" for q in meta.quantizations)
            + "\n"
        )

    body = meta.description.strip() or (
        f"Model exported from [Seiso Forge](https://github.com/seiso) by **{meta.author}**."
    )
    if meta.base_model:
        body += f"\n\nFine-tuned from `{meta.base_model}`."
    if meta.finetune_type:
        body += f"\n\n**Fine-tune type:** {meta.finetune_type}."

    provenance = []
    if meta.seiso_source:
        provenance.append(f"- **Source:** {meta.seiso_source}")
    if meta.seiso_job_id:
        provenance.append(f"- **Seiso job:** `{meta.seiso_job_id}`")
    if meta.export_formats:
        provenance.append(f"- **Export formats:** {', '.join(meta.export_formats)}")
    provenance.append(
        f"- **Exported:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    )
    if extra:
        for k, v in extra.items():
            provenance.append(f"- **{k}:** {v}")

    readme = (
        f"{_yaml_frontmatter(card)}\n\n"
        f"# {title}\n\n"
        f"{body}\n"
        f"{quant_line}\n"
        f"## Provenance\n\n" + "\n".join(provenance) + "\n\n## Usage\n\n"
        "### Transformers\n\n"
        "```python\n"
        "from transformers import AutoModelForCausalLM, AutoTokenizer\n\n"
        f'model = AutoModelForCausalLM.from_pretrained("{meta.repo_id}")\n'
        f'tokenizer = AutoTokenizer.from_pretrained("{meta.repo_id}")\n'
        "```\n\n"
        "### llama.cpp\n\n"
        "Use the GGUF artifacts in this repo with llama.cpp or the included Modelfile.\n"
    )
    return readme


def write_hub_artifacts(
    folder: Path, meta: HubModelMetadata, *, extra: dict[str, Any] | None = None
) -> dict[str, Path]:
    """Write README.md and seiso_model_metadata.json into an export folder before upload."""
    folder.mkdir(parents=True, exist_ok=True)
    readme = folder / "README.md"
    readme.write_text(render_readme(meta, extra=extra), encoding="utf-8")

    metadata_path = folder / "seiso_model_metadata.json"
    payload = {
        "repo_id": meta.repo_id,
        "author": meta.author,
        "license": meta.license,
        "base_model": meta.base_model,
        "finetune_type": meta.finetune_type,
        "export_formats": meta.export_formats,
        "quantizations": meta.quantizations,
        "tags": meta.tags,
        "seiso_job_id": meta.seiso_job_id,
        "seiso_source": meta.seiso_source,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"readme": readme, "metadata": metadata_path}


def metadata_from_manifest(
    meta: HubModelMetadata,
    manifest_path: Path,
) -> HubModelMetadata:
    """Enrich metadata from a training checkpoint seiso_manifest.json."""
    manifest = read_json_file(manifest_path, default={})
    if not isinstance(manifest, dict):
        return meta

    method = str(manifest.get("method", "")).lower()
    quant = str(manifest.get("quant", "")).lower()
    if method == "lora":
        meta.finetune_type = "qlora" if quant in {"4bit", "8bit"} else "lora"
    elif method == "full":
        meta.finetune_type = "full"
    elif method == "embedding":
        meta.finetune_type = "embedding"
        meta.pipeline_tag = "feature-extraction"
        meta.library_name = "sentence-transformers"
    elif method == "slime":
        meta.finetune_type = "slime"
    elif method == "nemo_rl":
        meta.finetune_type = "nemo_rl"

    if not meta.base_model:
        meta.base_model = manifest.get("model_id")
    return meta
