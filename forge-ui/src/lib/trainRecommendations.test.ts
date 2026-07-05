import { describe, expect, it } from "vitest";
import {
  pickAutoRecommendationFields,
  shouldAutoApplyRecommendationFields,
} from "./trainRecommendations";

describe("shouldAutoApplyRecommendationFields", () => {
  const rec = { dataset_format: "chatml", train_on_responses_only: true };

  it("applies when config is default and no dataset analysis", () => {
    expect(shouldAutoApplyRecommendationFields(false, rec, null)).toBe(true);
  });

  it("skips when user customized config", () => {
    expect(shouldAutoApplyRecommendationFields(true, rec, null)).toBe(false);
  });

  it("skips when dataset analysis is present", () => {
    expect(shouldAutoApplyRecommendationFields(false, rec, { format: "chatml" })).toBe(
      false,
    );
  });

  it("skips when recommendations are missing", () => {
    expect(shouldAutoApplyRecommendationFields(false, undefined, null)).toBe(false);
  });
});

describe("pickAutoRecommendationFields", () => {
  it("maps non-auto dataset format and train_on_responses_only", () => {
    expect(
      pickAutoRecommendationFields({
        dataset_format: "chatml",
        train_on_responses_only: true,
      }),
    ).toEqual({
      datasetFormat: "chatml",
      trainOnResponsesOnly: true,
    });
  });

  it("ignores auto dataset format", () => {
    expect(
      pickAutoRecommendationFields({
        dataset_format: "auto",
        train_on_responses_only: false,
      }),
    ).toEqual({ trainOnResponsesOnly: false });
  });

  it("returns empty object when nothing to apply", () => {
    expect(pickAutoRecommendationFields({ dataset_format: "auto" })).toEqual({});
  });
});
