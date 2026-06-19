import { request } from "./client";
import type { AuthUser } from "./types";

export const authApi = {
  authStatus: () => request<{ needs_onboarding: boolean }>("/auth/status"),
  register: (password: string) =>
    request<{ access_token: string; user: AuthUser }>("/auth/register", {
      method: "POST",
      body: JSON.stringify({ password }),
    }),
  login: (password: string) =>
    request<{ access_token: string; user: AuthUser }>("/auth/login", {
      method: "POST",
      body: JSON.stringify({ password }),
    }),
  me: () => request<AuthUser & { created_at: string }>("/auth/me"),
  logout: () => request<{ status: string }>("/auth/logout", { method: "POST" }),
};
