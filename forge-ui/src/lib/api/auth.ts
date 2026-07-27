import { request } from "./client";
import type { AuthUser } from "./types";

export type AuthResponse = {
  /** Opt-in Bearer JWT only; browser sessions use the HttpOnly cookie. */
  access_token?: string | null;
  user: AuthUser;
  /** Present only when Forge generated a fresh key during register. */
  nsec?: string | null;
};

export const authApi = {
  authStatus: () =>
    request<{
      needs_onboarding: boolean;
      storage_mode: "persistent" | "ephemeral";
      storage_mode_configured: boolean;
      auth_method: "nostr";
      /** Public owner identity when an account exists. */
      owner_npub?: string | null;
    }>("/auth/status"),
  register: (
    body?:
      | { generate?: true; nsec?: never }
      | { generate?: false; nsec: string },
    storageMode?: "persistent" | "ephemeral",
  ) =>
    request<AuthResponse>("/auth/register", {
      method: "POST",
      body: JSON.stringify({ ...(body || { generate: true }), storage_mode: storageMode }),
    }),
  login: (nsec: string) =>
    request<AuthResponse>("/auth/login", {
      method: "POST",
      body: JSON.stringify({ nsec }),
    }),
  resetSession: (confirmation: string) =>
    request<{
      status: string;
      needs_onboarding: boolean;
      sessions_rotated: boolean;
      rows_deleted: number;
    }>("/auth/reset-session", {
      method: "POST",
      body: JSON.stringify({ confirmation }),
    }),
  me: () => request<AuthUser & { created_at: string }>("/auth/me"),
  logout: () => request<{ status: string }>("/auth/logout", { method: "POST" }),
};
