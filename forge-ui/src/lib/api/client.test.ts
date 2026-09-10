import { afterEach, describe, expect, it, vi } from "vitest";
import { formatApiError, getCsrfToken, request } from "./client";

describe("API client", () => {
  afterEach(() => {
    vi.restoreAllMocks();
    document.cookie = "seiso_csrf=; Max-Age=0; path=/";
  });

  it("reads CSRF token from cookie", () => {
    document.cookie = "seiso_csrf=test-csrf-token; path=/";
    expect(getCsrfToken()).toBe("test-csrf-token");
  });

  it("sends CSRF header on mutating requests", async () => {
    document.cookie = "seiso_csrf=csrf-123; path=/";
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );

    await request<{ ok: boolean }>("/auth/logout", { method: "POST" });

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/auth/logout",
      expect.objectContaining({
        method: "POST",
        credentials: "include",
        headers: expect.objectContaining({ "X-CSRF-Token": "csrf-123" }),
      }),
    );
  });

  it("formats validation error arrays", () => {
    expect(formatApiError([{ msg: "Field required" }, { msg: "Too short" }])).toBe(
      "Field required; Too short",
    );
  });

  it("aborts a hung request after the per-call timeout", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(
      (_url: unknown, init?: RequestInit) =>
        new Promise<Response>((_resolve, reject) => {
          init?.signal?.addEventListener("abort", () =>
            reject(new DOMException("Aborted", "AbortError")),
          );
        }),
    );
    const pending = request<{ ok: boolean }>("/inference/models", {
      method: "GET",
      timeoutMs: 50,
    });
    const err = await pending.then(
      () => {
        throw new Error("expected request to reject on timeout");
      },
      (e: unknown) => e,
    );
    expect(err).toBeInstanceOf(DOMException);
    expect((err as DOMException).name).toBe("AbortError");
  });
});
