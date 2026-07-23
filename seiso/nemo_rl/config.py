"""Seiso-facing configuration for NVIDIA NeMo RL launches."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from seiso.compat import StrEnum


class NeMoRLRecipe(StrEnum):
    """Supported NeMo RL entrypoint recipes."""

    GRPO = "grpo"
    DPO = "dpo"
    DISTILLATION = "distillation"
    SMOKE = "smoke"  # GRPO smoke (10 steps / GSM8K) for install validation


_RECIPE_SCRIPTS: dict[str, str] = {
    NeMoRLRecipe.GRPO.value: "examples/run_grpo.py",
    NeMoRLRecipe.DPO.value: "examples/run_dpo.py",
    NeMoRLRecipe.DISTILLATION.value: "examples/run_distillation.py",
    NeMoRLRecipe.SMOKE.value: "examples/run_grpo.py",
}

_RECIPE_BASE_CONFIGS: dict[str, str] = {
    NeMoRLRecipe.GRPO.value: "examples/configs/grpo_math_1B.yaml",
    NeMoRLRecipe.DPO.value: "examples/configs/dpo.yaml",
    NeMoRLRecipe.DISTILLATION.value: "examples/configs/distillation_math.yaml",
    NeMoRLRecipe.SMOKE.value: "examples/configs/grpo_smoke.yaml",
}


@dataclass(frozen=True)
class NeMoRLConfig:
    """Launch config for an external NeMo RL run.

    Seiso does not reimplement NeMo RL algorithms. It projects a small set of
    knobs into Hydra overrides and shells out to ``uv run`` inside the NeMo RL
    checkout.
    """

    model_id: str
    output_dir: Path
    recipe: NeMoRLRecipe = NeMoRLRecipe.GRPO
    # Optional absolute path; else SEISO_NEMO_RL_ROOT / sibling discovery.
    nemo_rl_root: Path | None = None
    # Relative to nemo_rl_root; None → recipe default.
    base_config: str | None = None
    gpus_per_node: int = 1
    num_nodes: int = 1
    max_steps: int | None = None
    learning_rate: float | None = None
    # Map Seiso GRPO knobs when recipe is grpo/smoke.
    rollouts_per_prompt: int | None = None
    num_prompts_per_step: int | None = None
    seed: int = 42
    use_lora: bool = False
    # Extra Hydra-style overrides (e.g. "logger.wandb_enabled=False").
    extra_overrides: tuple[str, ...] = ()
    uv_executable: str | None = None
    # When true, write overlay + command but do not execute (tests / dry preview).
    dry_run: bool = False
    # Optional sandbox for path assertions (Forge user root).
    sandbox_root: Path | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not str(self.model_id or "").strip():
            raise ValueError("model_id is required for NeMo RL")
        if self.gpus_per_node < 1:
            raise ValueError("gpus_per_node must be >= 1")
        if self.num_nodes < 1:
            raise ValueError("num_nodes must be >= 1")
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError("max_steps must be >= 1 when set")
        if self.learning_rate is not None and self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0 when set")
        if self.rollouts_per_prompt is not None and self.rollouts_per_prompt < 1:
            raise ValueError("rollouts_per_prompt must be >= 1 when set")
        if self.num_prompts_per_step is not None and self.num_prompts_per_step < 1:
            raise ValueError("num_prompts_per_step must be >= 1 when set")
        recipe = self.recipe.value if isinstance(self.recipe, NeMoRLRecipe) else str(self.recipe)
        if recipe not in _RECIPE_SCRIPTS:
            raise ValueError(
                f"nemo_rl recipe must be one of: {', '.join(sorted(_RECIPE_SCRIPTS))} "
                f"(got {recipe!r})"
            )
        for ov in self.extra_overrides:
            if not str(ov).strip():
                raise ValueError("extra_overrides entries must be non-empty")
            if "\n" in str(ov) or "\x00" in str(ov):
                raise ValueError("extra_overrides entries must be single-line Hydra keys")

    def recipe_script(self) -> str:
        key = self.recipe.value if isinstance(self.recipe, NeMoRLRecipe) else str(self.recipe)
        return _RECIPE_SCRIPTS[key]

    def recipe_base_config(self) -> str:
        if self.base_config and str(self.base_config).strip():
            return str(self.base_config).strip()
        key = self.recipe.value if isinstance(self.recipe, NeMoRLRecipe) else str(self.recipe)
        return _RECIPE_BASE_CONFIGS[key]

    @classmethod
    def from_yaml(cls, path: str | Path) -> NeMoRLConfig:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError("NeMo RL config must be a mapping")
        return cls.from_mapping(data)

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> NeMoRLConfig:
        raw = dict(data)
        # Accept TrainConfig-style field names.
        aliases = {
            "nemo_rl_recipe": "recipe",
            "nemo_rl_base_config": "base_config",
            "nemo_rl_gpus_per_node": "gpus_per_node",
            "nemo_rl_num_nodes": "num_nodes",
            "nemo_rl_max_steps": "max_steps",
            "nemo_rl_use_lora": "use_lora",
            "nemo_rl_extra_overrides": "extra_overrides",
            "nemo_rl_dry_run": "dry_run",
            "rollout_batch_size": "num_prompts_per_step",
        }
        for src, dest in aliases.items():
            if src in raw and dest not in raw:
                raw[dest] = raw.pop(src)
            elif src in raw:
                raw.pop(src)

        recipe_raw = str(raw.get("recipe", NeMoRLRecipe.GRPO.value)).lower().strip()
        recipe = NeMoRLRecipe(recipe_raw)

        overrides = raw.get("extra_overrides") or ()
        if isinstance(overrides, str):
            overrides = tuple(
                part.strip() for part in overrides.split(",") if part.strip()
            )
        else:
            overrides = tuple(str(x) for x in overrides)

        cfg = cls(
            model_id=str(raw["model_id"]),
            output_dir=Path(raw.get("output_dir", "./outputs/nemo-rl")),
            recipe=recipe,
            nemo_rl_root=(
                Path(raw["nemo_rl_root"]) if raw.get("nemo_rl_root") else None
            ),
            base_config=raw.get("base_config"),
            gpus_per_node=int(raw.get("gpus_per_node", 1) or 1),
            num_nodes=int(raw.get("num_nodes", 1) or 1),
            max_steps=(
                int(raw["max_steps"]) if raw.get("max_steps") is not None else None
            ),
            learning_rate=(
                float(raw["learning_rate"])
                if raw.get("learning_rate") is not None
                else None
            ),
            rollouts_per_prompt=(
                int(raw["rollouts_per_prompt"])
                if raw.get("rollouts_per_prompt") is not None
                else None
            ),
            num_prompts_per_step=(
                int(raw["num_prompts_per_step"])
                if raw.get("num_prompts_per_step") is not None
                else None
            ),
            seed=int(raw.get("seed", 42) or 42),
            use_lora=bool(raw.get("use_lora", False)),
            extra_overrides=overrides,
            uv_executable=raw.get("uv_executable"),
            dry_run=bool(raw.get("dry_run", False)),
            sandbox_root=(
                Path(raw["sandbox_root"]) if raw.get("sandbox_root") else None
            ),
            extra=dict(raw.get("extra") or {}),
        )
        cfg.validate()
        return cfg
