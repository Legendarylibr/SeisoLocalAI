import { API, formatApiError, getCsrfToken, request } from "./client";
import type { KnowledgeBase, KnowledgeChunk } from "./types";

export const knowledgeApi = {
  listKnowledgeBases: () => request<{ bases: KnowledgeBase[] }>("/knowledge/bases"),
  createKnowledgeBase: (knowledge_base_id: string, name?: string) =>
    request<{ id: string; name: string; path: string }>("/knowledge/bases", {
      method: "POST",
      body: JSON.stringify({ knowledge_base_id, name }),
    }),
  uploadKnowledgeFile: async (file: File) => {
    const csrf = getCsrfToken();
    const headers: Record<string, string> = {};
    if (csrf) headers["X-CSRF-Token"] = csrf;
    const form = new FormData();
    form.append("file", file);
    const res = await fetch(`${API}/knowledge/upload`, {
      method: "POST",
      headers,
      body: form,
      credentials: "include",
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(formatApiError(err.detail, res.statusText));
    }
    return res.json() as Promise<{ path: string; filename: string; size: number }>;
  },
  ingestKnowledge: (knowledge_base_id: string, source_path: string) =>
    request<{ job_id: string; chunk_count?: number }>("/knowledge/ingest", {
      method: "POST",
      body: JSON.stringify({ knowledge_base_id, source_path }),
    }),
  retrieveKnowledge: (knowledge_base_id: string, query: string, top_k = 5) =>
    request<{ results: KnowledgeChunk[] }>("/knowledge/retrieve", {
      method: "POST",
      body: JSON.stringify({ knowledge_base_id, query, top_k }),
    }),
};
