import { afterEach, describe, expect, it, vi } from "vitest";
import { distillRlApi } from "./distillRl";

function mockJsonResponse(body: unknown) {
  return Promise.resolve(
    new Response(JSON.stringify(body), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    }),
  );
}

describe("distill-rl API payloads", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("loads presets from distill-rl route", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(() =>
      mockJsonResponse({
        presets: [{ id: "smoke", label: "Smoke", stages: ["distill", "rollout"] }],
        stages: ["distill", "rollout", "dpo", "evaluate"],
        help: { distill: "Teacher logits → student" },
      }),
    );

    const res = await distillRlApi.distillRLPresets();

    expect(fetchMock).toHaveBeenCalledWith("/api/distill-rl/presets", expect.any(Object));
    expect(res.presets[0]?.id).toBe("smoke");
    expect(res.stages).toContain("dpo");
  });

  it("starts distill-rl job with teacher, student, and multiseed", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(() =>
      mockJsonResponse({ job_id: "job-1", status: "pending" }),
    );

    await distillRlApi.startDistillRL({
      preset: "reproducible",
      teacher_model: "openai-community/gpt2",
      student_model: "openai-community/gpt2",
      seeds: [13, 42, 99],
      evaluate_teacher: true,
      stages: ["distill", "rollout", "dpo", "evaluate"],
    });

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/distill-rl/jobs",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          preset: "reproducible",
          teacher_model: "openai-community/gpt2",
          student_model: "openai-community/gpt2",
          seeds: [13, 42, 99],
          evaluate_teacher: true,
          stages: ["distill", "rollout", "dpo", "evaluate"],
        }),
      }),
    );
  });
});
