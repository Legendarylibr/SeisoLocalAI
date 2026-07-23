import { describe, expect, it } from "vitest";
import {
  getTrainingConfigBlockers,
  packingAllowedForFormat,
  packingConflictsWithResponseMask,
  preferenceRequiresOptIn,
  resolveEffectiveDatasetFormat,
} from "./trainingConfigGuards";

describe("trainingConfigGuards", () => {
  it("resolves auto to analysis format", () => {
    expect(resolveEffectiveDatasetFormat("auto", "preference")).toBe("preference");
    expect(resolveEffectiveDatasetFormat("chat", "preference")).toBe("chat");
  });

  it("flags packing + response-only on chat formats", () => {
    expect(packingConflictsWithResponseMask(true, true, "chat")).toBe(true);
    expect(packingConflictsWithResponseMask(true, true, "text")).toBe(false);
    expect(packingConflictsWithResponseMask(true, false, "chat")).toBe(false);
    expect(packingConflictsWithResponseMask(true, true, "auto")).toBe(true);
    expect(packingAllowedForFormat("preference", true)).toBe(false);
    expect(packingAllowedForFormat("text", true)).toBe(true);
    expect(packingAllowedForFormat("auto", true)).toBe(false);
  });

  it("requires preference opt-in", () => {
    expect(preferenceRequiresOptIn("preference", false)).toBe(true);
    expect(preferenceRequiresOptIn("preference", true)).toBe(false);
    expect(preferenceRequiresOptIn("chat", false)).toBe(false);
  });

  it("blocks preference + slime and packing conflicts", () => {
    const blockers = getTrainingConfigBlockers({
      method: "slime",
      datasetFormat: "auto",
      resolvedFormat: "preference",
      packing: true,
      trainOnResponsesOnly: true,
      preferenceAsSft: false,
      slimeDynamicSampling: false,
      slimeEvalDataset: "data/eval.jsonl",
    });
    expect(blockers.map((b) => b.code)).toEqual([
      "preference_not_for_slime",
      "packing_response_mask_conflict",
      "slime_needs_dynamic_sampling",
    ]);
  });

  it("blocks slime without held-out eval", () => {
    const blockers = getTrainingConfigBlockers({
      method: "slime",
      datasetFormat: "chat",
      packing: false,
      trainOnResponsesOnly: true,
      preferenceAsSft: false,
      slimeDynamicSampling: true,
      slimeEvalDataset: "",
    });
    expect(blockers.map((b) => b.code)).toEqual(["slime_needs_held_out_eval"]);
  });

  it("blocks preference without opt-in for LoRA", () => {
    const blockers = getTrainingConfigBlockers({
      method: "lora",
      datasetFormat: "preference",
      packing: false,
      trainOnResponsesOnly: true,
      preferenceAsSft: false,
    });
    expect(blockers.map((b) => b.code)).toEqual([
      "preference_needs_dpo_or_opt_in",
    ]);
  });

  it("blocks preference + nemo_rl grpo", () => {
    const blockers = getTrainingConfigBlockers({
      method: "nemo_rl",
      datasetFormat: "preference",
      packing: false,
      trainOnResponsesOnly: true,
      preferenceAsSft: false,
      nemoRlRecipe: "grpo",
    });
    expect(blockers.map((b) => b.code)).toEqual([
      "preference_not_for_nemo_rl_grpo",
    ]);
  });

  it("allows preference + nemo_rl dpo", () => {
    const blockers = getTrainingConfigBlockers({
      method: "nemo_rl",
      datasetFormat: "preference",
      packing: false,
      trainOnResponsesOnly: true,
      preferenceAsSft: false,
      nemoRlRecipe: "dpo",
    });
    expect(blockers.map((b) => b.code)).toEqual([]);
  });

  it("allows chosen-only SFT when opted in", () => {
    const blockers = getTrainingConfigBlockers({
      method: "lora",
      datasetFormat: "preference",
      packing: false,
      trainOnResponsesOnly: true,
      preferenceAsSft: true,
    });
    expect(blockers).toEqual([]);
  });
});
