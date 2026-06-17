const API = "/api";

export type AuthUser = { id: string; email: string; display_name: string | null };

export type SecurityPosture = {
  allow_tools: boolean;
  allow_code_exec: boolean;
  allow_openai_tools: boolean;
  allow_remote: boolean;
  autodefense_enabled: boolean;
  bind_localhost: boolean;
  db_encrypted: boolean;
  rate_limit: number;
  session_hours: number;
};

/** Read CSRF double-submit cookie set by the server on login/register. */
export function getCsrfToken(): string | null {
  const match = document.cookie.match(/(?:^|;\s*)seiso_csrf=([^;]*)/);
  return match ? decodeURIComponent(match[1]) : null;
}

/** Clear any legacy localStorage tokens from older builds. */
export function clearLegacyToken() {
  try {
    localStorage.removeItem("seiso_token");
  } catch {
    /* ignore */
  }
}

const MUTATING = new Set(["POST", "PUT", "DELETE", "PATCH"]);

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const method = (init.method || "GET").toUpperCase();
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...(init.headers as Record<string, string>),
  };
  if (MUTATING.has(method)) {
    const csrf = getCsrfToken();
    if (csrf) headers["X-CSRF-Token"] = csrf;
  }

  const res = await fetch(`${API}${path}`, { ...init, headers, credentials: "include" });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Request failed");
  }
  return res.json() as Promise<T>;
}

export const api = {
  authStatus: () => request<{ needs_onboarding: boolean }>("/auth/status"),
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
  listInferenceModels: () =>
    request<{ models: InferenceModelOption[]; total: number }>("/inference/models"),
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
  startExport: (checkpoint: string, formats: string[], hubRepo?: string, rlQuantJobId?: string) =>
    request<{ job_id: string }>("/export/jobs", {
      method: "POST",
      body: JSON.stringify({
        checkpoint,
        formats,
        hub_repo: hubRepo || null,
        rl_quant_job_id: rlQuantJobId || null,
      }),
    }),
  listRLQuantJobs: () => request<RLQuantJob[]>("/rl-quant/jobs"),
  rlQuantPresets: () =>
    request<{ presets: RLQuantPreset[]; reward_weights_help: Record<string, string> }>("/rl-quant/presets"),
  startRLQuant: (body: Record<string, unknown>) =>
    request<{ job_id: string; status: string }>("/rl-quant/jobs", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  listCompressJobs: () => request<CompressJob[]>("/compress/jobs"),
  compressPresets: () =>
    request<{ presets: CompressPreset[]; stages: string[]; help: Record<string, string> }>(
      "/compress/presets",
    ),
  startCompress: (body: Record<string, unknown>) =>
    request<{ job_id: string; status: string }>("/compress/jobs", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  listImageCompressJobs: () => request<ImageCompressJob[]>("/image-compress/jobs"),
  imageCompressPresets: () =>
    request<{ presets: ImageCompressPreset[]; stages: string[]; help: Record<string, string> }>(
      "/image-compress/presets",
    ),
  startImageCompress: (body: Record<string, unknown>) =>
    request<{ job_id: string; status: string }>("/image-compress/jobs", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  settings: () =>
    request<{
      host: string;
      port: number;
      data_dir: string;
      backend: string;
      allow_remote: boolean;
      hf_configured: boolean;
      autodefense_enabled: boolean;
      autodefense_configured: boolean;
      security: SecurityPosture;
    }>("/settings"),
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

export type InferenceModelOption = {
  id: string;
  kind: "local" | "ollama";
  name: string;
  source: string;
  source_label: string;
  format: string | null;
  default_backend: string;
  backends: string[];
  backend_labels: Record<string, string>;
  ollama_model: string | null;
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

export type RLQuantJob = {
  id: string;
  status: string;
  config_json: string;
  output_dir: string | null;
  recommendation_path: string | null;
  gguf_quants: string[];
  created_at: string;
};

export type RLQuantPreset = {
  id: string;
  label: string;
  backend: string;
  training_backend: string;
};

export type CompressJob = {
  id: string;
  status: string;
  config_json: string;
  output_dir: string | null;
  run_dir: string | null;
  model_dir: string | null;
  stages: string[];
  stage_results: Record<string, unknown>;
  created_at: string;
};

export type CompressPreset = {
  id: string;
  label: string;
  stages: string[];
};

export type ImageCompressJob = {
  id: string;
  status: string;
  config_json: string;
  output_dir: string | null;
  run_dir: string | null;
  model_dir: string | null;
  stages: string[];
  stage_results: Record<string, unknown>;
  created_at: string;
};

export type ImageCompressPreset = {
  id: string;
  label: string;
  stages: string[];
};

/** Stream chat completions via SSE (cookie session + CSRF). */
export async function streamChat(
  body: Record<string, unknown>,
  handlers: {
    onEvent: (event: string, data: string) => void;
    onError?: (message: string) => void;
  },
): Promise<void> {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  const csrf = getCsrfToken();
  if (csrf) headers["X-CSRF-Token"] = csrf;

  const res = await fetch(`${API}/inference/chat`, {
    method: "POST",
    headers,
    credentials: "include",
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Chat request failed");
  }

  const reader = res.body?.getReader();
  if (!reader) return;
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const blocks = buffer.split("\n\n");
    buffer = blocks.pop() || "";
    for (const block of blocks) {
      let event = "message";
      let data = "";
      for (const line of block.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        if (line.startsWith("data:")) data = line.slice(5).trim();
      }
      if (data) handlers.onEvent(event, data);
    }
  }
}

export function subscribeSSE(path: string, onEvent: (event: string, data: string) => void): () => void {
  const controller = new AbortController();
  fetch(`${API}${path}`, { credentials: "include", signal: controller.signal }).then(async (res) => {
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
