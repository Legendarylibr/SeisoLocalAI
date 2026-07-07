import { request } from "./client";
import { streamPostSSE } from "./sse";
import { cachedGet } from "./getCache";
import type {
  ChatContextStatus,
  ChatMessage,
  ChatThread,
  HardwareSummary,
  InferenceModelOption,
  ModelVariantsResponse,
} from "./types";

export const inferenceApi = {
  listInferenceModels: () =>
    cachedGet<{
      models: InferenceModelOption[];
      total: number;
      hardware_summary?: HardwareSummary;
      preferred_inference_backend?: string;
      local_only?: boolean;
      model_router?: { enabled: boolean; url: string; model_id: string };
    }>("/inference/models", 60_000),
  routerStatus: () =>
    request<{
      enabled: boolean;
      health?: Record<string, unknown>;
      detail?: Record<string, unknown>;
    }>("/inference/router/status"),
  streamPreloadModel: (
    model_id: string,
    inference_backend: string,
    handlers: {
      onProgress: (data: Record<string, unknown>) => void;
      onComplete: (data: Record<string, unknown>) => void;
      onError?: (message: string) => void;
    },
    options?: { max_tokens?: number; n_ctx?: number | null },
  ) =>
    streamPostSSE(
      "/inference/preload/stream",
      {
        model_id,
        inference_backend,
        ...(options?.max_tokens != null ? { max_tokens: options.max_tokens } : {}),
        ...(options?.n_ctx != null ? { n_ctx: options.n_ctx } : {}),
      },
      {
        progress: (data) => handlers.onProgress(JSON.parse(data)),
        complete: (data) => handlers.onComplete(JSON.parse(data)),
        error: (data) => handlers.onError?.(data),
      },
    ),
  cancelInference: () => request<{ active_model: string | null }>("/inference/cancel", { method: "POST" }),
  cancelGeneration: () => request<{ active_model: string | null }>("/inference/cancel-generation", { method: "POST" }),
  listThreads: () => request<ChatThread[]>("/inference/threads"),
  createThread: (title: string, model_id?: string) =>
    request<ChatThread>("/inference/threads", {
      method: "POST",
      body: JSON.stringify({ title, model_id }),
    }),
  getMessages: (threadId: string) =>
    request<ChatMessage[]>(`/inference/threads/${threadId}/messages`),
  getContextStatus: (params: {
    thread_id?: string | null;
    max_tokens?: number;
    n_ctx?: number | null;
    tools?: boolean;
    knowledge_base_id?: string | null;
    model_id?: string | null;
    draft_message?: string | null;
  }) => {
    const q = new URLSearchParams();
    if (params.thread_id) q.set("thread_id", params.thread_id);
    if (params.max_tokens != null) q.set("max_tokens", String(params.max_tokens));
    if (params.n_ctx != null) q.set("n_ctx", String(params.n_ctx));
    if (params.tools) q.set("tools", "true");
    if (params.knowledge_base_id) q.set("knowledge_base_id", params.knowledge_base_id);
    if (params.model_id) q.set("model_id", params.model_id);
    if (params.draft_message) q.set("draft_message", params.draft_message);
    const suffix = q.toString();
    return request<ChatContextStatus>(`/inference/context${suffix ? `?${suffix}` : ""}`);
  },
  deleteThread: (id: string) => request<{ status: string }>(`/inference/threads/${id}`, { method: "DELETE" }),
  getModelVariants: (modelId: string) =>
    cachedGet<ModelVariantsResponse>(`/inference/models/${encodeURIComponent(modelId)}/variants`, 30_000),
};
