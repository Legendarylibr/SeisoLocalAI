import { request } from "./client";
import type { RLQuantJob, RLQuantPreset } from "./types";

export const rlQuantApi = {
  listRLQuantJobs: () => request<RLQuantJob[]>("/rl-quant/jobs"),
  rlQuantPresets: () =>
    request<{
      presets: RLQuantPreset[];
      preset_hints?: Record<string, string>;
      reward_weights_help: Record<string, string>;
      kernel_rl_help?: Record<string, string | string[]>;
    }>("/rl-quant/presets"),
  startRLQuant: (body: Record<string, unknown>) =>
    request<{ job_id: string; status: string }>("/rl-quant/jobs", {
      method: "POST",
      body: JSON.stringify(body),
    }),
};
