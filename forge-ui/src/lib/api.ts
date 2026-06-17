const API = "/api";

export type AuthUser = { id: string; email: string; display_name: string | null };

let token: string | null = localStorage.getItem("seiso_token");

export function setToken(t: string | null) {
  token = t;
  if (t) localStorage.setItem("seiso_token", t);
  else localStorage.removeItem("seiso_token");
}

export function getToken() {
  return token;
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...(init.headers as Record<string, string>),
  };
  if (token) headers.Authorization = `Bearer ${token}`;

  const res = await fetch(`${API}${path}`, { ...init, headers, credentials: "include" });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Request failed");
  }
  return res.json() as Promise<T>;
}

export const api = {
  authStatus: () => request<{ needs_onboarding: boolean; user_count: number }>("/auth/status"),
  register: (email: string, password: string, display_name?: string) =>
    request<{ access_token: string; user: AuthUser }>("/auth/register", {
      method: "POST",
      body: JSON.stringify({ email, password, display_name }),
    }),
  login: (email: string, password: string) =>
    request<{ access_token: string; user: AuthUser }>("/auth/login", {
      method: "POST",
      body: JSON.stringify({ email, password }),
    }),
  me: () => request<AuthUser & { created_at: string }>("/auth/me"),
  logout: () => request<{ status: string }>("/auth/logout", { method: "POST" }),
  listModels: () => request<LocalModel[]>("/models"),
  catalog: (q = "", family?: string, task?: string) => {
    const params = new URLSearchParams();
    if (q) params.set("q", q);
    if (family) params.set("family", family);
    if (task) params.set("task", task);
    return request<{ models: CatalogModel[]; families: string[]; total: number }>(
      `/models/catalog?${params}`,
    );
  },
  downloadModel: (repo_id: string, filename?: string, variant: "auto" | "safetensors" | "gguf" = "auto") =>
    request<{ downloaded: string[] }>("/models/download", {
      method: "POST",
      body: JSON.stringify({ repo_id, filename, variant: variant === "auto" ? "auto" : variant }),
    }),
  vramStatus: () => request<{ active_model: string | null; backend: string | null }>("/models/vram"),
  unloadVram: () => request<{ active_model: string | null }>("/models/vram/unload", { method: "POST" }),
  listThreads: () => request<ChatThread[]>("/inference/threads"),
  createThread: (title: string, model_id?: string) =>
    request<ChatThread>("/inference/threads", {
      method: "POST",
      body: JSON.stringify({ title, model_id }),
    }),
  getMessages: (threadId: string) =>
    request<ChatMessage[]>(`/inference/threads/${threadId}/messages`),
  startTraining: (config: Record<string, unknown>, multi_gpu = false) =>
    request<{ job_id: string; status: string }>("/training/jobs", {
      method: "POST",
      body: JSON.stringify({ config, multi_gpu }),
    }),
  listTrainingJobs: () => request<TrainingJob[]>("/training/jobs"),
  startExport: (checkpoint: string, formats: string[], hubRepo?: string) =>
    request<{ job_id: string }>("/export/jobs", {
      method: "POST",
      body: JSON.stringify({ checkpoint, formats, hub_repo: hubRepo || null }),
    }),
  settings: () =>
    request<{ host: string; port: number; data_dir: string; backend: string }>("/settings"),
  runRecipe: (recipe: Record<string, unknown>) =>
    request<{ job_id: string }>("/recipes/jobs", {
      method: "POST",
      body: JSON.stringify({ recipe }),
    }),
  listProviders: () => request<Array<{ id: string; name: string; provider_type: string; config: Record<string, unknown> }>>("/providers"),
  createProvider: (body: { name: string; provider_type: string; config: Record<string, unknown> }) =>
    request("/providers", { method: "POST", body: JSON.stringify(body) }),
  deleteProvider: (id: string) => request(`/providers/${id}`, { method: "DELETE" }),
  listMcpServers: () =>
    request<Array<{ id: string; name: string; command: string; args: string[]; enabled: boolean }>>("/mcp/servers"),
  createMcpServer: (body: { name: string; command: string; args: string[] }) =>
    request("/mcp/servers", { method: "POST", body: JSON.stringify(body) }),
  connectMcp: (id: string) => request<{ connected: boolean; tools: string[] }>(`/mcp/servers/${id}/connect`, { method: "POST" }),
  deleteMcpServer: (id: string) => request(`/mcp/servers/${id}`, { method: "DELETE" }),
};

export type LocalModel = {
  id: string;
  name: string;
  path: string;
  source: string | null;
  format: string | null;
  size_bytes: number;
};

export type CatalogModel = {
  repo_id: string;
  name: string;
  family: string;
  params: string;
  task: string;
  quant: string;
  tags: string[];
};

export type ChatThread = { id: string; title: string; model_id: string | null; created_at: string };
export type ChatMessage = { id: string; role: string; content: string; created_at: string };
export type TrainingJob = {
  id: string;
  status: string;
  config_json: string;
  created_at: string;
};

export function subscribeSSE(path: string, onEvent: (event: string, data: string) => void): () => void {
  const url = `${API}${path}`;
  const headers: Record<string, string> = {};
  if (token) headers.Authorization = `Bearer ${token}`;

  const controller = new AbortController();
  fetch(url, { headers, credentials: "include", signal: controller.signal }).then(async (res) => {
    const reader = res.body?.getReader();
    if (!reader) return;
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split("\n\n");
      buffer = parts.pop() || "";
      for (const block of parts) {
        let event = "message";
        let data = "";
        for (const line of block.split("\n")) {
          if (line.startsWith("event:")) event = line.slice(6).trim();
          if (line.startsWith("data:")) data = line.slice(5).trim();
        }
        if (data) onEvent(event, data);
      }
    }
  });
  return () => controller.abort();
}
