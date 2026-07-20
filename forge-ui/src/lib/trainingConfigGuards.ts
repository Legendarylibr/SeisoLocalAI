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
  return Boolean(packing && trainOnResponsesOnly && isChatStyleFormat(format));
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
};

export type TrainingConfigBlocker = {
  code:
    | "preference_needs_dpo_or_opt_in"
    | "preference_not_for_slime"
    | "packing_response_mask_conflict"
    | "slime_needs_dynamic_sampling";
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

  if (format === "preference" && input.method === "slime") {
    blockers.push({
      code: "preference_not_for_slime",
      message:
        "SLIME GRPO needs verifiable prompt/answer rows, not preference pairs. Use Distill-RL for DPO, or LoRA/full with chosen-only SFT.",
    });
  } else if (preferenceRequiresOptIn(format, input.preferenceAsSft)) {
    blockers.push({
      code: "preference_needs_dpo_or_opt_in",
      message:
        "Preference pairs need Distill-RL/DPO for real alignment, or enable “Train on chosen only (not DPO)” to run chosen-response SFT.",
    });
  }

  if (
    packingConflictsWithResponseMask(
      input.packing,
      input.trainOnResponsesOnly,
      format,
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

  return blockers;
}

export function packingAllowedForFormat(
  format: string,
  trainOnResponsesOnly: boolean,
): boolean {
  if (!trainOnResponsesOnly) return true;
  return !isChatStyleFormat(format);
}
