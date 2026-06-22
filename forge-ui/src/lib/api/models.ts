import { API, request } from "./client";
import { streamPostSSE } from "./sse";
import type { CatalogModel, HardwareSummary, LocalModel, VramStatus } from "./types";

export const modelsApi = {
  listModels: () => request<LocalModel[]>("/models"),
  vramStatus: () => request<VramStatus>("/models/vram"),
  freeMemory: () => request<VramStatus>("/models/vram/unload", { method: "POST" }),
  downloadLocalModel: (modelId: string) =>
    fetch(`${API}/models/${modelId}/download`, { credentials: "include" }),
  catalog: (q = "", family?: string, task?: string, fitsOnly = false, cursor?: string | null, limit = 50, purpose: "chat" | "train" = "chat") => {
    const params = new URLSearchParams({ hardware_aware: "true", limit: String(limit), purpose });
    if (q) params.set("q", q);
    if (family) params.set("family", family);
    if (task) params.set("task", task);
    if (fitsOnly) params.set("fits_only", "true");
    if (cursor) params.set("cursor", cursor);
    return request<{
      models: CatalogModel[];
      families: string[];
      total: number;
      limit: number;
      next_cursor: string | null;
      has_more: boolean;
      source: string;
      hardware_summary?: HardwareSummary;
      local_only?: boolean;
    }>(`/models/catalog?${params}`);
  },
  streamDownloadModel: (
    repo_id: string,
    handlers: {
      onProgress: (data: Record<string, unknown>) => void;
      onComplete: (data: Record<string, unknown>) => void;
      onError?: (message: string) => void;
    },
    variant: "auto" | "safetensors" | "gguf" = "gguf",
    options: { filename?: string; revision?: string } = {},
  ) =>
    streamPostSSE(
      "/models/download/stream",
      {
        repo_id,
        variant: variant === "auto" ? "auto" : variant,
        ...(options.filename ? { filename: options.filename } : {}),
        ...(options.revision ? { revision: options.revision } : {}),
      },
      {
        progress: (data) => handlers.onProgress(JSON.parse(data)),
        complete: (data) => handlers.onComplete(JSON.parse(data)),
        error: (data) => handlers.onError?.(data),
      },
    ),
};
