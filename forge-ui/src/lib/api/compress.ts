import { request } from "./client";
import type { CompressJob, CompressPreset } from "./types";

export const compressApi = {
  listCompressJobs: () => request<CompressJob[]>("/compress/jobs"),
  compressPresets: () =>
    request<{ presets: CompressPreset[]; stages: string[]; help: Record<string, string>; defaults?: Record<string, string> }>(
      "/compress/presets",
    ),
  startCompress: (body: Record<string, unknown>) =>
    request<{ job_id: string; status: string }>("/compress/jobs", {
      method: "POST",
      body: JSON.stringify(body),
    }),
};
