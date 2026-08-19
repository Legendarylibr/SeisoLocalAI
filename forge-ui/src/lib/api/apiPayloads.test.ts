import { afterEach, describe, expect, it, vi } from "vitest";
import { authApi } from "./auth";
import { exportApi } from "./export";

function mockJsonResponse(body: unknown) {
  return Promise.resolve(new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  }));
}

describe("API payloads", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("sends onboarding storage mode when registering", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(() =>
      mockJsonResponse({
        user: { id: "u1", display_name: "Admin", npub: "npub1test" },
        nsec: "nsec1test",
      }),
    );

    await authApi.register({ generate: true }, "persistent");

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/auth/register",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ generate: true, storage_mode: "persistent" }),
      }),
    );
  });

  it("passes GGUF quantizations through export start", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(() =>
      mockJsonResponse({ job_id: "job-1" }),
    );

    await exportApi.startExport(
      "/tmp/checkpoint",
      ["gguf"],
      undefined,
      undefined,
      ["q4_k_m", "q8_0"],
    );

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/export/jobs",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          checkpoint: "/tmp/checkpoint",
          formats: ["gguf"],
          profile: null,
          gguf_quantizations: ["q4_k_m", "q8_0"],
          hub: null,
        }),
      }),
    );
  });

  it("passes GGUF quantizations through export precheck", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(() =>
      mockJsonResponse({
        ok: true,
        repo_id: "user/model",
        errors: [],
        warnings: [],
        model_card_preview: "",
      }),
    );

    await exportApi.precheckHubExport({
      hub: { username: "user", model_name: "model", author: "User" },
      formats: ["gguf"],
      gguf_quantizations: ["q5_k_m"],
    });

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/export/precheck",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          hub: { username: "user", model_name: "model", author: "User" },
          formats: ["gguf"],
          gguf_quantizations: ["q5_k_m"],
        }),
      }),
    );
  });
});
