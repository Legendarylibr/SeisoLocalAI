import { describe, expect, it } from "vitest";
import { computeTokensPerSec, estimateOutputTokens, formatTokensPerSec } from "./streamSpeed";

describe("streamSpeed", () => {
  it("estimates tokens from output text", () => {
    expect(estimateOutputTokens("")).toBe(0);
    expect(estimateOutputTokens("hello world")).toBeGreaterThan(0);
  });

  it("computes tok/s from elapsed time", () => {
    expect(computeTokensPerSec(100, 2000)).toBe(50);
    expect(computeTokensPerSec(0, 1000)).toBeNull();
  });

  it("formats tok/s for display", () => {
    expect(formatTokensPerSec(42.37)).toBe("42.4 tok/s");
    expect(formatTokensPerSec(128.4)).toBe("128 tok/s");
  });
});
