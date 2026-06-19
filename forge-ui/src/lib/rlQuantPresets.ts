import type { RLQuantPreset } from "@/lib/api";

export const RL_QUANT_FALLBACK_PRESETS: RLQuantPreset[] = [
  { id: "minimal", label: "Fast smoke (256 episodes)", backend: "simulator", training_backend: "stdlib" },
  { id: "reproducible", label: "Reproducible research (simulator)", backend: "simulator", training_backend: "stdlib" },
  { id: "post_train", label: "Post fine-tune RL (continuous, router)", backend: "simulator", training_backend: "stdlib" },
];

export const RL_QUANT_PRESET_HINTS: Record<string, string> = {
  minimal: "Fast smoke run — simulator backend, few episodes.",
  reproducible: "Fixed seeds and logged artifacts for paper-grade reproducibility.",
  post_train: "Post fine-tune checkpoint — links training output to quant recommendation.",
};
