import { request } from "./client";

export type ProviderRow = {
  id: string;
  name: string;
  provider_type: string;
  config: Record<string, unknown>;
  created_at?: string;
};

export type ManagedVllmStatus = {
  running?: boolean;
  managed?: boolean;
  enabled?: boolean;
  feature_enabled?: boolean;
  cloud_multigpu_enabled?: boolean;
  autostart?: boolean;
  vllm_available?: boolean;
  suggested_tensor_parallel?: number;
  gpu_count?: number;
  model?: string;
  base_url?: string;
  tensor_parallel_size?: number;
  pid?: number | null;
  healthy?: boolean;
  log_path?: string;
};

export const providersApi = {
  listProviders: () => request<ProviderRow[]>("/providers"),
  createProvider: (body: {
    name: string;
    provider_type: string;
    config: Record<string, unknown>;
  }) => request<ProviderRow>("/providers", { method: "POST", body: JSON.stringify(body) }),
  deleteProvider: (id: string) => request(`/providers/${id}`, { method: "DELETE" }),
  managedVllmStatus: () =>
    request<ManagedVllmStatus>("/providers/managed-vllm/status"),
  managedVllmPreview: (body: Record<string, unknown>) =>
    request<Record<string, unknown>>("/providers/managed-vllm/preview", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  managedVllmStart: (body: Record<string, unknown>) =>
    request<{
      status: ManagedVllmStatus;
      provider: ProviderRow | null;
      compat?: { base_url?: string; model_ids?: string[]; note?: string };
    }>("/providers/managed-vllm/start", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  managedVllmStop: () =>
    request<Record<string, unknown>>("/providers/managed-vllm/stop", {
      method: "POST",
    }),
};
