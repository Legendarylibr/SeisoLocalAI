import { describe, expect, it } from "vitest";
import {
  computeTokensPerSec,
  estimateOutputTokens,
  formatTokensPerSec,
  parseStreamStats,
  resolveOutputTokenCount,
} from "./streamSpeed";

describe("streamSpeed", () => {
  it("parses server stream stats", () => {
    expect(parseStreamStats('{"output_tokens": 42}')).toEqual({ output_tokens: 42 });
    expect(parseStreamStats("not-json")).toBeNull();
    expect(parseStreamStats('{"output_tokens": -1}')).toBeNull();
    expect(
      parseStreamStats(
        '{"output_tokens": 100, "finish_reason": "length", "truncated": true, "auto_continues": 2}',
      ),
    ).toEqual({
      output_tokens: 100,
      finish_reason: "length",
      truncated: true,
      auto_continues: 2,
    });
  });

  it("prefers measured token counts with text fallback", () => {
    expect(resolveOutputTokenCount(12, "")).toBe(12);
    expect(resolveOutputTokenCount(0, "hello world")).toBeGreaterThan(0);
  });

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
