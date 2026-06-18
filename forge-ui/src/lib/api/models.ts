import { API, request } from "./client";
import { streamPostSSE } from "./sse";
import type { CatalogModel, HardwareSummary, LocalModel } from "./types";

export const modelsApi = {
  listModels: () => request<LocalModel[]>("/models"),
  downloadLocalModel: (modelId: string) =>
    fetch(`${API}/models/${modelId}/download`, { credentials: "include" }),
  catalog: (q = "", family?: string, task?: string, fitsOnly = false) => {
    const params = new URLSearchParams({ hardware_aware: "true" });
    if (q) params.set("q", q);
    if (family) params.set("family", family);
    if (task) params.set("task", task);
    if (fitsOnly) params.set("fits_only", "true");
    return request<{
      models: CatalogModel[];
      families: string[];
      total: number;
      hardware_summary?: HardwareSummary;
      local_only?: boolean;
    }>(`/models/catalog?${params}`);
  },
  downloadModel: (repo_id: string, filename?: string, variant: "auto" | "safetensors" | "gguf" = "auto") =>
    request<{ downloaded: string[]; repo_id: string; gguf_repo?: string; variant: string; model_id: string; cache_dir: string }>(
      "/models/download",
      {
        method: "POST",
        body: JSON.stringify({ repo_id, filename, variant: variant === "auto" ? "auto" : variant }),
      },
    ),
  streamDownloadModel: (
    repo_id: string,
    handlers: {
      onProgress: (data: Record<string, unknown>) => void;
      onComplete: (data: Record<string, unknown>) => void;
      onError?: (message: string) => void;
    },
    variant: "auto" | "safetensors" | "gguf" = "gguf",
  ) =>
    streamPostSSE(
      "/models/download/stream",
      { repo_id, variant: variant === "auto" ? "auto" : variant },
      {
        progress: (data) => handlers.onProgress(JSON.parse(data)),
        complete: (data) => handlers.onComplete(JSON.parse(data)),
        error: (data) => handlers.onError?.(data),
      },
    ),
};
