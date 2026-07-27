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
  nostrStatus: () =>
    cachedGet<{
      server_allow_nostr: boolean;
      key_saved: boolean;
      npub: string | null;
      auto_attest: boolean;
      relays: string[];
      allow_loopback: boolean;
    }>("/settings/nostr", 15_000),
  saveNostrPrefs: async (body: {
    auto_attest: boolean;
    relays: string[];
    allow_loopback: boolean;
  }) => {
    const res = await request<Record<string, unknown>>("/settings/nostr", {
      method: "PUT",
      body: JSON.stringify(body),
    });
    invalidateApiCache("/settings/nostr");
    return res;
  },
  nostrKeygen: async () => {
    const res = await request<{ status: string; npub: string; nsec?: string }>(
      "/settings/nostr/keygen",
      {
        method: "POST",
      },
    );
    invalidateApiCache("/settings/nostr");
    return res;
  },
  importNostrKey: async (secret: string) => {
    const res = await request<{ status: string; npub: string }>("/settings/nostr/key", {
      method: "PUT",
      body: JSON.stringify({ secret }),
    });
    invalidateApiCache("/settings/nostr");
    return res;
  },
  clearNostrKey: async () => {
    const res = await request<{ status: string }>("/settings/nostr/key", { method: "DELETE" });
    invalidateApiCache("/settings/nostr");
    return res;
  },
};
