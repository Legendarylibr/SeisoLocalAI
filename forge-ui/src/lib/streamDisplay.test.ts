import { afterEach, describe, expect, it, vi } from "vitest";
import { createStreamDisplaySink } from "./streamDisplay";

describe("createStreamDisplaySink", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("paints the first chunk on a microtask", async () => {
    const paints: string[] = [];
    const sink = createStreamDisplaySink((text) => paints.push(text));
    sink.push("hello");
    expect(paints).toEqual([]);
    await Promise.resolve();
    expect(paints).toEqual(["hello"]);
  });

  it("coalesces follow-up updates into smooth catch-up frames", async () => {
    const paints: Array<{ text: string; continuing?: boolean }> = [];
    const rafQueue: FrameRequestCallback[] = [];
    vi.spyOn(window, "requestAnimationFrame").mockImplementation((cb: FrameRequestCallback) => {
      rafQueue.push(cb);
      return rafQueue.length;
    });

    const sink = createStreamDisplaySink((text, state) => {
      paints.push({ text, continuing: state?.continuing });
    });
    sink.push("a");
    await Promise.resolve();
    expect(paints[0]?.text).toBe("a");

    // Large multi-pass resume batch — should not dump entire string in one paint.
    sink.push("a" + "x".repeat(200));
    expect(rafQueue.length).toBeGreaterThan(0);
    rafQueue.shift()?.(0);
    const afterFirstCatchup = paints[paints.length - 1]?.text ?? "";
    expect(afterFirstCatchup.length).toBeGreaterThan(1);
    expect(afterFirstCatchup.length).toBeLessThan(201);

    // Drain frames until caught up.
    let guard = 0;
    while (rafQueue.length && guard < 40) {
      rafQueue.shift()?.(0);
      guard += 1;
    }
    expect(paints[paints.length - 1]?.text).toBe("a" + "x".repeat(200));
  });

  it("noteContinue keeps text and signals continuing state", async () => {
    const paints: Array<{ text: string; continuing?: boolean }> = [];
    vi.spyOn(window, "requestAnimationFrame").mockImplementation((cb: FrameRequestCallback) => {
      cb(0);
      return 1;
    });
    const sink = createStreamDisplaySink((text, state) => {
      paints.push({ text, continuing: state?.continuing });
    });
    sink.push("Part one");
    await Promise.resolve();
    sink.noteContinue();
    const last = paints[paints.length - 1];
    expect(last?.text).toBe("Part one");
    expect(last?.continuing).toBe(true);
  });

  it("flush forces the full server draft through", () => {
    const paints: string[] = [];
    vi.spyOn(window, "requestAnimationFrame").mockReturnValue(1);
    const sink = createStreamDisplaySink((text) => paints.push(text));
    sink.push("one");
    sink.push("one two three");
    sink.flush();
    expect(paints[paints.length - 1]).toBe("one two three");
  });
});
