import { request } from "./client";
import type { AuthUser } from "./types";

export const authApi = {
  authStatus: () =>
    request<{
      needs_onboarding: boolean;
      storage_mode: "persistent" | "ephemeral";
      storage_mode_configured: boolean;
    }>("/auth/status"),
  register: (password: string, storageMode?: "persistent" | "ephemeral") =>
    request<{ access_token: string; user: AuthUser }>("/auth/register", {
      method: "POST",
      body: JSON.stringify({ password, storage_mode: storageMode }),
    }),
  login: (password: string) =>
    request<{ access_token: string; user: AuthUser }>("/auth/login", {
      method: "POST",
      body: JSON.stringify({ password }),
    }),
  me: () => request<AuthUser & { created_at: string }>("/auth/me"),
  logout: () => request<{ status: string }>("/auth/logout", { method: "POST" }),
};
