import { request } from "./client";
import { streamPostSSE } from "./sse";
import { cachedGet } from "./getCache";
import type { ChatMessage, ChatThread, HardwareSummary, InferenceModelOption } from "./types";

export const inferenceApi = {
  listInferenceModels: () =>
    cachedGet<{
      models: InferenceModelOption[];
      total: number;
      hardware_summary?: HardwareSummary;
      preferred_inference_backend?: string;
      local_only?: boolean;
    }>("/inference/models", 60_000),
  streamPreloadModel: (
    model_id: string,
    inference_backend: string,
    handlers: {
      onProgress: (data: Record<string, unknown>) => void;
      onComplete: (data: Record<string, unknown>) => void;
      onError?: (message: string) => void;
    },
  ) =>
    streamPostSSE(
      "/inference/preload/stream",
      { model_id, inference_backend },
      {
        progress: (data) => handlers.onProgress(JSON.parse(data)),
        complete: (data) => handlers.onComplete(JSON.parse(data)),
        error: (data) => handlers.onError?.(data),
      },
    ),
  cancelInference: () => request<{ active_model: string | null }>("/inference/cancel", { method: "POST" }),
  listThreads: () => request<ChatThread[]>("/inference/threads"),
  createThread: (title: string, model_id?: string) =>
    request<ChatThread>("/inference/threads", {
      method: "POST",
      body: JSON.stringify({ title, model_id }),
    }),
  getMessages: (threadId: string) =>
    request<ChatMessage[]>(`/inference/threads/${threadId}/messages`),
  deleteThread: (id: string) => request<{ status: string }>(`/inference/threads/${id}`, { method: "DELETE" }),
};
