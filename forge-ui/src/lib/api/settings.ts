import { request } from "./client";
import type { HfAuthInfo, HfHubStatus, SecurityPosture } from "./types";

export const settingsApi = {
  settings: () =>
    request<{
      host: string;
      port: number;
      data_dir: string;
      backend: string;
      allow_remote: boolean;
      hf_configured: boolean;
      hf_auth: HfAuthInfo;
      autodefense_enabled: boolean;
      autodefense_configured: boolean;
      security: SecurityPosture;
    }>("/settings"),
  saveHfToken: (token: string) =>
    request<{ status: string }>("/settings/hf-token", {
      method: "PUT",
      body: JSON.stringify({ token }),
    }),
  clearHfToken: () => request<{ status: string }>("/settings/hf-token", { method: "DELETE" }),
  hfStatus: () => request<HfHubStatus>("/settings/hf-status"),
};
