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
  token_invalid?: boolean;
};

export type HfHubStatus = {
  auth: HfAuthInfo & { token_source: string; token_invalid?: boolean };
  connectivity: {
    reachable: boolean;
    latency_ms: number | null;
    token_valid: boolean;
    token_invalid?: boolean;
    token_username: string | null;
    anonymous_ok: boolean;
    error: string | null;
    warning?: string | null;
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
  ready_for_local_chat?: boolean;
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
  memory_load_blocked?: boolean;
  memory_load_blocked_reason?: string | null;
  metadata?: Record<string, unknown>;
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
  download_available?: boolean;
  download_mirror_verified?: boolean;
  download_error?: string;
  gguf_repo?: string;
  gguf_file?: string;
  hardware_fit?: "ideal" | "good" | "tight" | "unlikely";
  hardware_fit_label?: string;
  hardware_note?: string;
  est_vram_mb?: number;
  memory_load_blocked?: boolean;
  memory_load_blocked_reason?: string | null;
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
