import { request } from "./client";
import type { DistillRLJob, DistillRLPreset } from "./types";

export const distillRlApi = {
  listDistillRLJobs: () => request<DistillRLJob[]>("/distill-rl/jobs"),
  distillRLPresets: () =>
    request<{
      presets: DistillRLPreset[];
      stages: string[];
      help: Record<string, string>;
      defaults?: Record<string, string>;
    }>("/distill-rl/presets"),
  startDistillRL: (body: Record<string, unknown>) =>
    request<{ job_id: string; status: string }>("/distill-rl/jobs", {
      method: "POST",
      body: JSON.stringify(body),
    }),
};
