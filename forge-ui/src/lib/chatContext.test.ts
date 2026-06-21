import { describe, expect, it } from "vitest";
import { formatContextLabel, formatTokenCount } from "./chatContext";

describe("chatContext", () => {
  it("formats token counts compactly", () => {
    expect(formatTokenCount(850)).toBe("850");
    expect(formatTokenCount(3200)).toBe("3.2k");
    expect(formatTokenCount(12000)).toBe("12k");
  });

  it("formats context labels", () => {
    expect(formatContextLabel("auto")).toBe("Auto");
    expect(formatContextLabel(4096)).toBe("4.1k");
  });
});
