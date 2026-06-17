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
  rate_limit_enabled: boolean;
  session_hours: number;
};

/** Read CSRF double-submit cookie set by the server on login/register. */
function getCsrfToken(): string | null {
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

function formatApiError(detail: unknown, fallback = "Request failed"): string {
  if (typeof detail === "string" && detail.trim()) return detail;
  if (Array.isArray(detail)) {
    const parts = detail
      .map((item) => {
        if (typeof item === "string") return item;
        if (item && typeof item === "object" && "msg" in item) {
          return String((item as { msg?: unknown }).msg ?? "");
        }
        return "";
      })
      .filter(Boolean);
    if (parts.length) return parts.join("; ");
  }
  if (detail && typeof detail === "object" && "msg" in detail) {
    return String((detail as { msg?: unknown }).msg ?? fallback);
  }
  return fallback;
}

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
    throw new Error(formatApiError(err.detail, res.statusText || "Request failed"));
  }
  return res.json() as Promise<T>;
}

export const api = {
  authStatus: () => request<{ needs_onboarding: boolean }>("/auth/status"),
  register: (password: string) => request<{ access_token: string; user: AuthUser }>("/auth/register", {
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
  listModels: () => request<LocalModel[]>("/models"),
  downloadLocalModel: (modelId: string) =>
    fetch(`${API}/models/${modelId}/download`, { credentials: "include" }),
  listPublishableOutputs: () =>
    request<PublishableModel[]>("/export/publishable"),
  listExportProfiles: () =>
    request<{ id: string; formats: string[]; default_gguf_quants: string[] }[]>("/export/profiles"),
  precheckHubExport: (body: {
    hub: HubPublishFields;
    formats?: string[];
    profile?: string;
    gguf_quantizations?: string[];
  }) =>
    request<{
      ok: boolean;
      repo_id: string;
      errors: string[];
      warnings: string[];
      model_card_preview: string;
    }>("/export/precheck", { method: "POST", body: JSON.stringify(body) }),
  listExportJobs: () => request<ExportJob[]>("/export/jobs"),
  listInferenceModels: () =>
    request<{
      models: InferenceModelOption[];
      total: number;
      hardware_summary?: HardwareSummary;
      preferred_inference_backend?: string;
      local_only?: boolean;
    }>("/inference/models"),
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
  startTraining: (
    config: Record<string, unknown>,
    multi_gpu = false,
    export_on_complete?: {
      profile?: string;
      formats?: string[];
      gguf_quantizations?: string[];
    },
  ) =>
    request<{ job_id: string; status: string }>("/training/jobs", {
      method: "POST",
      body: JSON.stringify({ config, multi_gpu, export_on_complete }),
    }),
  listTrainingJobs: () => request<TrainingJob[]>("/training/jobs"),
  listTrainingModels: () => request<{ models: TrainableModel[]; total: number }>("/training/models"),
  searchDatasets: (q: string, limit = 12) => {
    const params = new URLSearchParams({ q, limit: String(limit) });
    return request<{ datasets: CatalogDataset[]; total: number }>(`/training/datasets?${params}`);
  },
  getTrainingMetrics: (jobId: string) => request<TrainingMetricsPayload>(`/training/jobs/${jobId}/metrics`),
  startExport: (
    checkpoint: string,
    formats: string[],
    hub?: HubPublishFields,
    rlQuantJobId?: string,
    profile?: string,
  ) =>
    request<{ job_id: string }>("/export/jobs", {
      method: "POST",
      body: JSON.stringify({
        checkpoint,
        formats,
        profile: profile || null,
        hub: hub || null,
        rl_quant_job_id: rlQuantJobId || null,
      }),
    }),
  publishToHub: (body: {
    model_id?: string;
    export_job_id?: string;
    output_path?: string;
    hub: HubPublishFields;
  }) =>
    request<{ repo_id: string; path: string; log: string }>("/export/publish", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  downloadExportOutput: (jobId: string, key = "gguf") =>
    fetch(`${API}/export/outputs/${jobId}/download?key=${encodeURIComponent(key)}`, {
      credentials: "include",
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
      hf_auth: HfAuthInfo;
      autodefense_enabled: boolean;
      autodefense_configured: boolean;
      security: SecurityPosture;
    }>("/settings"),
  saveHfToken: (token: string) =>
    request<{ status: string }>("/settings/hf-token", {
      method: "PUT",
      body: JSON.stringify({ token }),
    }),
  clearHfToken: () => request<{ status: string }>("/settings/hf-token", { method: "DELETE" }),
  hfStatus: () =>
    request<HfHubStatus>("/settings/hf-status"),
  runRecipe: (recipe: Record<string, unknown>) =>
    request<{ job_id: string }>("/recipes/jobs", {
      method: "POST",
      body: JSON.stringify({ recipe }),
    }),
  listProviders: () => request<Array<{ id: string; name: string; provider_type: string; config: Record<string, unknown> }>>("/providers"),
  createProvider: (body: { name: string; provider_type: string; config: Record<string, unknown> }) =>
    request("/providers", { method: "POST", body: JSON.stringify(body) }),
  deleteProvider: (id: string) => request(`/providers/${id}`, { method: "DELETE" }),
  deleteThread: (id: string) => request<{ status: string }>(`/inference/threads/${id}`, { method: "DELETE" }),
  hardware: () => request<HardwareProfile>("/system/hardware"),
  metrics: () => request<SystemMetrics>("/system/metrics"),
  guide: (goal: string) =>
    request<{ goal: string; steps: GuideStep[]; hardware_summary: Record<string, unknown>; local_only: boolean }>(
      `/system/guide?goal=${goal}`,
    ),
  listKnowledgeBases: () =>
    request<{ bases: KnowledgeBase[] }>("/knowledge/bases"),
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

export type LocalModel = {
  id: string;
  name: string;
  path: string;
  source: string | null;
  format: string | null;
  size_bytes: number;
  pushable?: boolean;
};

export type PublishableModel = {
  id: string;
  name: string;
  path: string;
  source: string | null;
  format: string | null;
  size_bytes: number;
  job_id?: string;
  export_key?: string;
};

export type ExportJob = {
  id: string;
  status: string;
  config_json: string;
  output_paths_json?: string;
  created_at: string;
};

export type HubPublishFields = {
  username: string;
  model_name: string;
  author: string;
  license?: string;
  base_model?: string;
  description?: string;
  tags?: string[];
  hf_token?: string;
  use_cli?: boolean;
};

export type HfAuthInfo = {
  cli_available: boolean;
  cli_binary: string | null;
  cli_logged_in: boolean;
  token_configured: boolean;
  token_sources: string[];
  user_token_saved: boolean;
};

export type HfHubStatus = {
  auth: HfAuthInfo & { token_source: string };
  connectivity: {
    reachable: boolean;
    latency_ms: number | null;
    token_valid: boolean;
    token_username: string | null;
    anonymous_ok: boolean;
    error: string | null;
  };
  transfer: {
    backend: string;
    xet_available: boolean;
    xet_version: string | null;
    high_performance: boolean;
    num_threads: string;
    download_timeout_s: string;
    hints: string[];
    hint: string | null;
  };
  cache_dir: string;
  runtime: {
    llamacpp: boolean;
    mlx: boolean;
    torch: boolean;
    huggingface_hub: boolean;
    install_hints: string[];
  };
  ready_for_download: boolean;
  ready_for_gguf_chat: boolean;
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
  hardware_fit?: "ideal" | "good" | "tight" | "unlikely";
  hardware_fit_label?: string;
  hardware_note?: string;
  est_vram_mb?: number;
};

export type CatalogDataset = {
  repo_id: string;
  name: string;
  downloads?: number | null;
  tags: string[];
};

export type CatalogModel = {
  repo_id: string;
  name: string;
  family: string;
  params: string;
  task: string;
  quant: string;
  tags: string[];
  featured?: boolean;
  priority?: number;
  download_bytes?: number;
  download_bytes_estimated?: boolean;
  gguf_repo?: string;
  gguf_file?: string;
  hardware_fit?: "ideal" | "good" | "tight" | "unlikely";
  hardware_fit_label?: string;
  hardware_note?: string;
  est_vram_mb?: number;
};

export type TrainingDefaults = {
  batch_size: number;
  gradient_accumulation_steps: number;
  max_seq_length: number;
  quant: string;
  method: string;
  gradient_checkpointing: boolean;
  max_recommended_params: string;
  use_fused_kernels?: boolean;
  use_fused_ce?: boolean;
  kernel_backend?: string;
  train_platform?: string;
  multi_gpu_available?: boolean;
  note: string;
};

export type HardwareSummary = {
  tier: string;
  tier_label: string;
  backend: string;
  ram_gb: number;
  gpu_count: number;
  effective_vram_mb: number;
  vram_headroom_mb: number;
  memory_headroom_label?: string;
  preferred_inference_backend: string;
  preferred_inference_backend_label?: string;
  local_only: boolean;
};

export type HardwareProfile = {
  platform: string;
  arch: string;
  backend: string;
  cpu_cores: number;
  cpu_brand: string;
  ram_gb: number;
  disk_free_gb: number;
  gpus: Array<{
    name: string;
    vram_total_mb: number | null;
    vram_used_mb: number | null;
    utilization_pct: number | null;
    temperature_c: number | null;
  }>;
  local_only: boolean;
  privacy: string;
  tier?: string;
  tier_label?: string;
  effective_vram_mb?: number;
  vram_headroom_mb?: number;
  preferred_inference_backend?: string;
  training_defaults?: TrainingDefaults;
  recommended_chat_repo?: string | null;
  recommended_train_repo?: string | null;
};

export type SystemMetrics = {
  cpu_util_pct: number | null;
  cpu_temp_c: number | null;
  ram_used_pct: number;
  gpus: Array<{
    name: string;
    vram_total_mb: number | null;
    vram_used_mb: number | null;
    utilization_pct: number | null;
    temperature_c: number | null;
  }>;
  local_only: boolean;
  ts: number;
};

export type GuideStep = { title: string; detail: string; path: string };

export type KnowledgeBase = {
  id: string;
  chunk_count: number;
  has_index: boolean;
};

export type KnowledgeChunk = {
  id: string;
  text: string;
  source: string;
  chunk_index: number;
};

export type ChatThread = { id: string; title: string; model_id: string | null; created_at: string };
export type ChatMessage = { id: string; role: string; content: string; created_at: string };
export type TrainingJob = {
  id: string;
  status: string;
  config_json: string;
  metrics_json?: string;
  created_at: string;
};

export type TrainableModel = {
  id: string;
  name: string;
  path: string;
  repo_id: string | null;
  source: string | null;
  format: string;
  size_bytes: number;
};

export type TrainingMetricPoint = {
  type?: "training" | "eval" | "system";
  step?: number;
  epoch?: number;
  loss?: number | null;
  eval_loss?: number | null;
  reward?: number | null;
  learning_rate?: number | null;
  grad_norm?: number | null;
  train_samples_per_second?: number | null;
  train_steps_per_second?: number | null;
  ts?: number;
};

export type TrainingMetricsPayload = {
  summary: Record<string, unknown>;
  training: TrainingMetricPoint[];
  system: SystemMetrics[];
  updated_at?: number | null;
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

/** Stream chat completions via SSE (cookie session + CSRF). Returns abort handle. */
export function streamChat(
  body: Record<string, unknown>,
  handlers: {
    onEvent: (event: string, data: string) => void;
    onError?: (message: string) => void;
  },
): { promise: Promise<void>; abort: () => void } {
  const controller = new AbortController();
  const promise = (async () => {
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    const csrf = getCsrfToken();
    if (csrf) headers["X-CSRF-Token"] = csrf;

    let res: Response;
    try {
      res = await fetch(`${API}/inference/chat`, {
        method: "POST",
        headers,
        credentials: "include",
        body: JSON.stringify(body),
        signal: controller.signal,
      });
    } catch (err) {
      if (controller.signal.aborted) return;
      throw err;
    }

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || "Chat request failed");
    }

    const reader = res.body?.getReader();
    if (!reader) return;
    const decoder = new TextDecoder();
    let buffer = "";

    try {
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
    } catch (err) {
      if (!controller.signal.aborted) throw err;
    } finally {
      reader.cancel().catch(() => {});
    }
  })();

  return {
    promise,
    abort: () => {
      controller.abort();
      api.cancelInference().catch(() => {});
    },
  };
}

/** Stream SSE from a POST endpoint (cookie session + CSRF). Returns abort handle. */
function streamPostSSE(
  path: string,
  body: Record<string, unknown>,
  handlers: Record<string, (data: string) => void>,
): { promise: Promise<void>; abort: () => void } {
  const controller = new AbortController();
  const promise = (async () => {
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    const csrf = getCsrfToken();
    if (csrf) headers["X-CSRF-Token"] = csrf;

    let res: Response;
    try {
      res = await fetch(`${API}${path}`, {
        method: "POST",
        headers,
        credentials: "include",
        body: JSON.stringify(body),
        signal: controller.signal,
      });
    } catch (err) {
      if (controller.signal.aborted) return;
      throw err;
    }

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(formatApiError(err.detail, res.statusText || "Request failed"));
    }

    const reader = res.body?.getReader();
    if (!reader) throw new Error("Streaming response unavailable");
    const decoder = new TextDecoder();
    let buffer = "";

    try {
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
          if (data && handlers[event]) handlers[event](data);
        }
      }
    } catch (err) {
      if (!controller.signal.aborted) throw err;
    } finally {
      reader.cancel().catch(() => {});
    }
  })();

  return { promise, abort: () => controller.abort() };
}

export function subscribeSSE(
  path: string,
  onEvent: (event: string, data: string) => void,
  onError?: (err: Error) => void,
): () => void {
  const controller = new AbortController();

  void (async () => {
    let res: Response;
    try {
      res = await fetch(`${API}${path}`, { credentials: "include", signal: controller.signal });
    } catch (err) {
      if (!controller.signal.aborted) {
        onError?.(err instanceof Error ? err : new Error("SSE connection failed"));
      }
      return;
    }

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      onError?.(new Error(formatApiError(err.detail, res.statusText || "SSE request failed")));
      return;
    }

    const reader = res.body?.getReader();
    if (!reader) {
      onError?.(new Error("SSE stream unavailable"));
      return;
    }

    const decoder = new TextDecoder();
    let buffer = "";
    try {
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
    } catch (err) {
      if (!controller.signal.aborted) {
        onError?.(err instanceof Error ? err : new Error("SSE stream failed"));
      }
    } finally {
      reader.cancel().catch(() => {});
    }
  })();

  return () => controller.abort();
}
