"""SeisoModel — load, LoRA, and export via Seiso Core loaders and export pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from seiso.models.loader import LoadOptions, ModelKind, load_model
from seiso.models.lora_targets import get_lora_target_modules, modules_exist_in_model

logger = logging.getLogger(__name__)


def resolve_dtype(dtype: str | None = None) -> str | None:
    """Pick bfloat16 when CUDA supports it."""
    if dtype:
        return dtype
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return "bfloat16"
    except ImportError:
        pass
    return None


@dataclass
class SeisoModelConfig:
    model_id: str
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    dtype: str | None = None
    trust_remote_code: bool = False
    use_flash_attention: bool = True


class SeisoModel:
    """High-level training handle: load → attach LoRA → train → export merged/GGUF."""

    def __init__(self, model: Any, tokenizer: Any, *, max_seq_length: int) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        *,
        max_seq_length: int = 2048,
        dtype: str | None = None,
        load_in_4bit: bool = True,
        load_in_8bit: bool = False,
        trust_remote_code: bool = False,
        use_flash_attention: bool = True,
    ) -> SeisoModel:
        dtype = resolve_dtype(dtype)
        opts = LoadOptions(
            model_id=model_name,
            kind=ModelKind.TEXT,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            max_seq_length=max_seq_length,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            use_flash_attention=use_flash_attention,
        )
        model, tokenizer = load_model(opts, for_training=True)
        logger.info("Loaded %s", model_name)
        return cls(model, tokenizer, max_seq_length=max_seq_length)

    @classmethod
    def attach_lora(
        cls,
        model: Any,
        *,
        r: int = 16,
        target_modules: list[str] | None = None,
        lora_alpha: int | None = None,
        lora_dropout: float = 0.0,
        bias: Literal["none", "all", "lora_only"] = "none",
        use_gradient_checkpointing: bool = True,
        use_rslora: bool = False,
        model_id: str = "",
    ) -> Any:
        """Apply LoRA adapters via PEFT."""
        from peft import LoraConfig, TaskType, get_peft_model

        if target_modules is None:
            target_modules = get_lora_target_modules(model_id, model)
        target_modules = modules_exist_in_model(model, target_modules)

        if use_gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={
                "use_reentrant": False
            })
            if hasattr(model.config, "use_cache"):
                model.config.use_cache = False

        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha or r * 2,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias=bias,
            task_type=TaskType.CAUSAL_LM,
            use_rslora=use_rslora,
        )
        return get_peft_model(model, lora_config)

    @staticmethod
    def for_training(model: Any) -> Any:
        """Prepare model for training."""
        model.train()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        return model

    @staticmethod
    def for_inference(model: Any) -> Any:
        """Prepare model for inference."""
        model.eval()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = True
        return model

    def export_merged(self, save_directory: str | Path, *, safe_serialization: bool = True) -> Path:
        """Merge LoRA into base weights and write a Hugging Face checkpoint."""
        dest = Path(save_directory)
        dest.mkdir(parents=True, exist_ok=True)

        from peft import PeftModel

        if isinstance(self.model, PeftModel):
            merged = self.model.merge_and_unload()
            merged.save_pretrained(str(dest), safe_serialization=safe_serialization)
        else:
            self.model.save_pretrained(str(dest), safe_serialization=safe_serialization)
        self.tokenizer.save_pretrained(str(dest))
        return dest

    def export_gguf(
        self,
        save_directory: str | Path,
        *,
        quantization_method: str | list[str] = "q4_k_m",
    ) -> list[Path]:
        """Export GGUF quant(s) and Modelfile via Seiso's export pipeline."""
        from seiso.export.gguf import export_gguf

        quants = (
            [quantization_method]
            if isinstance(quantization_method, str)
            else list(quantization_method)
        )
        return export_gguf(self.model, self.tokenizer, Path(save_directory), quants)
