import { request } from "./client";

export const providersApi = {
  listProviders: () =>
    request<Array<{ id: string; name: string; provider_type: string; config: Record<string, unknown> }>>("/providers"),
  createProvider: (body: { name: string; provider_type: string; config: Record<string, unknown> }) =>
    request("/providers", { method: "POST", body: JSON.stringify(body) }),
  deleteProvider: (id: string) => request(`/providers/${id}`, { method: "DELETE" }),
};
