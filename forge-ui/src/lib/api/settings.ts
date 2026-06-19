import { request } from "./client";
import { cachedGet, invalidateApiCache } from "./getCache";
import type { HfAuthInfo, HfHubStatus, SecurityPosture } from "./types";

export const settingsApi = {
  settings: () =>
    cachedGet<{
      host: string;
      port: number;
      data_dir: string;
      training_backend: string;
      inference_backends: string[];
      allow_remote: boolean;
      hf_configured: boolean;
      hf_auth: HfAuthInfo;
      security: SecurityPosture;
    }>("/settings", 300_000),
  saveHfToken: async (token: string) => {
    const res = await request<{ status: string }>("/settings/hf-token", {
      method: "PUT",
      body: JSON.stringify({ token }),
    });
    invalidateApiCache("/settings");
    return res;
  },
  clearHfToken: async () => {
    const res = await request<{ status: string }>("/settings/hf-token", { method: "DELETE" });
    invalidateApiCache("/settings");
    return res;
  },
  hfStatus: () => cachedGet<HfHubStatus>("/settings/hf-status", 15_000),
};
