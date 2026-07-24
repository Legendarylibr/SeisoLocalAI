/** Client-side training config honesty — mirrors Seiso TrainConfig / slime validators. */

export const CHAT_STYLE_FORMATS = [
  "chat",
  "alpaca",
  "sharegpt",
  "preference",
] as const;

export type ChatStyleFormat = (typeof CHAT_STYLE_FORMATS)[number];

export function isChatStyleFormat(format: string): boolean {
  return (CHAT_STYLE_FORMATS as readonly string[]).includes(format);
}

export function resolveEffectiveDatasetFormat(
  selectedFormat: string,
  resolvedFormat?: string | null,
): string {
  if (selectedFormat && selectedFormat !== "auto") return selectedFormat;
  if (resolvedFormat && resolvedFormat !== "auto") return resolvedFormat;
  return selectedFormat || "auto";
}

export function packingConflictsWithResponseMask(
  packing: boolean,
  trainOnResponsesOnly: boolean,
  format: string,
): boolean {
  // Treat unresolved "auto" like chat-style — server rejects packing+response-mask
  // for auto until format resolves (mirrors TrainConfig packing_blocked_formats).
  const chatLike = isChatStyleFormat(format) || format === "auto";
  return Boolean(packing && trainOnResponsesOnly && chatLike);
}

export function preferenceRequiresOptIn(
  format: string,
  preferenceAsSft: boolean,
): boolean {
  return format === "preference" && !preferenceAsSft;
}

export type TrainingStudioGuardInput = {
  method: string;
  datasetFormat: string;
  resolvedFormat?: string | null;
  packing: boolean;
  trainOnResponsesOnly: boolean;
  preferenceAsSft: boolean;
  slimeDynamicSampling?: boolean;
  /** Held-out verifiable eval JSONL (required for product slime). */
  slimeEvalDataset?: string;
  /** When true, materialize may auto-split held-out (data_gen + source). */
  slimeMaterializeSplitsEval?: boolean;
  /** NeMo RL recipe when method=nemo_rl (grpo | dpo | distillation | smoke). */
  nemoRlRecipe?: string;
};

export type TrainingConfigBlocker = {
  code:
    | "preference_needs_dpo_or_opt_in"
    | "preference_not_for_slime"
    | "preference_not_for_nemo_rl_grpo"
    | "packing_response_mask_conflict"
    | "slime_needs_dynamic_sampling"
    | "slime_needs_held_out_eval";
  message: string;
};

/** Return blockers that must be cleared before Start training. */
export function getTrainingConfigBlockers(
  input: TrainingStudioGuardInput,
): TrainingConfigBlocker[] {
  const format = resolveEffectiveDatasetFormat(
    input.datasetFormat,
    input.resolvedFormat,
  );
  const blockers: TrainingConfigBlocker[] = [];
  const nemoRecipe = (input.nemoRlRecipe || "grpo").toLowerCase();

  if (format === "preference" && input.method === "slime") {
    blockers.push({
      code: "preference_not_for_slime",
      message:
        "SLIME GRPO needs verifiable prompt/answer rows, not preference pairs. Use Distill-RL for DPO, or LoRA/full with chosen-only SFT.",
    });
  } else if (
    format === "preference" &&
    input.method === "nemo_rl" &&
    nemoRecipe !== "dpo"
  ) {
    blockers.push({
      code: "preference_not_for_nemo_rl_grpo",
      message:
        "NeMo RL GRPO/smoke need verifiable prompts. Switch recipe to DPO, open Distill-RL, or use chosen-only SFT.",
    });
  } else if (
    format === "preference" &&
    input.method === "nemo_rl" &&
    nemoRecipe === "dpo"
  ) {
    // NeMo RL DPO consumes preference pairs — no Seiso preference_as_sft opt-in.
  } else if (preferenceRequiresOptIn(format, input.preferenceAsSft)) {
    blockers.push({
      code: "preference_needs_dpo_or_opt_in",
      message:
        "Preference pairs need Distill-RL/DPO for real alignment, or enable “Train on chosen only (not DPO)” to run chosen-response SFT.",
    });
  }

  // Packing conflict must use the *posted* format. Start still sends
  // dataset_format: "auto" even when analysis resolved to text — server
  // Treats auto as packing-blocked, so prefer the selected format here.
  const packingFormat =
    input.datasetFormat === "auto" ? "auto" : format;
  if (
    packingConflictsWithResponseMask(
      input.packing,
      input.trainOnResponsesOnly,
      packingFormat,
    )
  ) {
    blockers.push({
      code: "packing_response_mask_conflict",
      message:
        "Sequence packing cannot be combined with train-on-responses-only for chat-style datasets. Use packing only for plain text, or turn off response-only loss.",
    });
  }

  if (input.method === "slime" && input.slimeDynamicSampling === false) {
    blockers.push({
      code: "slime_needs_dynamic_sampling",
      message:
        "SLIME GRPO needs reward-diverse groups (dynamic sampling). Turning it off makes advantages vacuous.",
    });
  }

  if (
    input.method === "slime" &&
    !input.slimeMaterializeSplitsEval &&
    !(input.slimeEvalDataset || "").trim()
  ) {
    blockers.push({
      code: "slime_needs_held_out_eval",
      message:
        "SLIME product runs need a held-out eval JSONL (distinct from train) for verifiable metrics.",
    });
  }

  return blockers;
}

export function packingAllowedForFormat(
  format: string,
  trainOnResponsesOnly: boolean,
): boolean {
  if (!trainOnResponsesOnly) return true;
  if (format === "auto") return false;
  return !isChatStyleFormat(format);
}
