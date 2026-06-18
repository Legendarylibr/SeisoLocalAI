import { request } from "./client";
import type { ImageCompressJob, ImageCompressPreset } from "./types";

export const imageCompressApi = {
  listImageCompressJobs: () => request<ImageCompressJob[]>("/image-compress/jobs"),
  imageCompressPresets: () =>
    request<{ presets: ImageCompressPreset[]; stages: string[]; help: Record<string, string>; defaults?: Record<string, string> }>(
      "/image-compress/presets",
    ),
  startImageCompress: (body: Record<string, unknown>) =>
    request<{ job_id: string; status: string }>("/image-compress/jobs", {
      method: "POST",
      body: JSON.stringify(body),
    }),
};
