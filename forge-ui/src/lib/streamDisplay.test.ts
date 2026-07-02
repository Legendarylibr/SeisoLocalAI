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

  it("coalesces follow-up updates into a single rAF paint", async () => {
    const paints: string[] = [];
    const rafCb = { current: null as FrameRequestCallback | null };
    vi.spyOn(window, "requestAnimationFrame").mockImplementation((cb: FrameRequestCallback) => {
      rafCb.current = cb;
      return 1;
    });

    const sink = createStreamDisplaySink((text) => paints.push(text));
    sink.push("a");
    await Promise.resolve();
    expect(paints).toEqual(["a"]);

    sink.push("ab");
    sink.push("abc");
    expect(paints).toEqual(["a"]);
    expect(rafCb.current).not.toBeNull();
    rafCb.current?.(0);
    expect(paints).toEqual(["a", "abc"]);
  });

  it("flush forces the latest draft through", () => {
    const paints: string[] = [];
    vi.spyOn(window, "requestAnimationFrame").mockReturnValue(1);
    const sink = createStreamDisplaySink((text) => paints.push(text));
    sink.push("one");
    sink.flush();
    expect(paints).toEqual(["one"]);
  });
});