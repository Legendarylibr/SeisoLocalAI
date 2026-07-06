export type AuthUser = { id: string; email: string; display_name: string | null };

export type SecurityPosture = {
  allow_tools: boolean;
  allow_code_exec: boolean;
  allow_openai_tools: boolean;
  allow_remote: boolean;
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
    llamacpp_error?: string | null;
  };
  ready_for_download: boolean;
  ready_for_upload: boolean;
  ready_for_gguf_chat: boolean;
  ready_for_local_chat?: boolean;
};

export const ROUTER_MODEL_ID = "__seiso_router__";

export type InferenceModelOption = {
  id: string;
  kind: "local" | "router";
  name: string;
  source: string;
  source_label: string;
  format: string | null;
  default_backend: string;
  backends: string[];
  backend_labels: Record<string, string>;
  size_bytes: number;
  selectable?: boolean;
  status?: "ready" | "incomplete" | string;
  status_note?: string;
  context_ceiling?: number;
  architecture?: string | null;
  is_moe?: boolean;
  uses_swa?: boolean;
  hardware_fit?: "ideal" | "good" | "tight" | "unlikely";
  hardware_fit_label?: string;
  hardware_note?: string;
  est_vram_mb?: number;
  memory_load_blocked?: boolean;
  memory_load_blocked_reason?: string | null;
  install_hints?: string[];
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

export type VramStatus = {
  local: {
    active_model: string | null;
    path?: string | null;
    backend?: string | null;
  };
  active_model: string | null;
  headroom_mb: number;
  memory_label: string;
  ram_gb?: number;
  apple_unified?: boolean;
  tier?: string;
  memory_profile?: "low" | "balanced";
  recommended_max_chat?: string | null;
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
  inference_backend_labels?: Record<string, string>;
  training_defaults?: TrainingDefaults;
  recommended_chat_repo?: string | null;
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

export type ChatContextStatus = {
  char_budget: number;
  char_used: number;
  char_total: number;
  message_count: number;
  messages_included: number;
  messages_omitted: number;
  estimated_prompt_tokens: number;
  max_tokens: number;
  n_ctx: number;
  n_ctx_auto: number;
  n_ctx_min: number;
  n_ctx_max: number;
  context_tokens_used: number;
  context_tokens_limit: number;
  fill_ratio: number;
  history_trimmed: boolean;
  context_window_options?: number[];
};

export type InferenceLocalVariant = {
  id: string;
  name?: string | null;
  quant: string;
  size_bytes?: number;
  path?: string | null;
  hardware_fit?: InferenceModelOption["hardware_fit"];
  hardware_fit_label?: string;
  memory_load_blocked?: boolean;
  selected?: boolean;
  source: "local";
  repo_id?: string | null;
  gguf_file?: string | null;
};

export type InferenceHubVariant = {
  quant: string;
  gguf_file: string;
  gguf_repo: string;
  source: "hub";
  downloaded: boolean;
  local_id?: string | null;
  selected?: boolean;
  size_bytes?: number;
  hardware_fit?: InferenceModelOption["hardware_fit"];
  hardware_fit_label?: string;
  memory_load_blocked?: boolean;
};

export type InferenceDraftCandidate = {
  id: string;
  name?: string | null;
  size_bytes?: number;
  format?: string | null;
  backends?: string[];
  hardware_fit?: InferenceModelOption["hardware_fit"];
  hardware_fit_label?: string;
};

export type ModelVariantsResponse = {
  model_id: string;
  variant_group?: string | null;
  gguf_repo?: string | null;
  catalog_repo?: string | null;
  base_model?: string | null;
  current_quant?: string | null;
  local_variants: InferenceLocalVariant[];
  hub_variants: InferenceHubVariant[];
  draft_candidates: InferenceDraftCandidate[];
  supports_speculative: boolean;
  supports_llamacpp: boolean;
};

export type TrainingJob = {
  id: string;
  status: string;
  config_json: string;
  metrics_json?: string;
  created_at: string;
};

export type DatasetLengthStats = {
  chars_min: number;
  chars_p50: number;
  chars_p95: number;
  chars_max: number;
  estimated_tokens_p95: number;
};

export type DatasetAnalysis = {
  valid: boolean;
  analysis_token?: string;
  dataset: string;
  split: string;
  columns: string[];
  initial_samples: number;
  kept: number;
  removed_invalid: number;
  removed_duplicate: number;
  utilization_pct: number;
  resolved_format: string;
  format_confidence: number;
  domain: string;
  domain_label: string;
  length_stats: DatasetLengthStats;
  recommended_config: {
    dataset_format: string;
    train_on_responses_only: boolean;
    preprocess_dataset: boolean;
    deduplicate_dataset: boolean;
    max_seq_length: number;
    epochs: number;
    early_stopping: boolean;
    early_stopping_patience: number;
    packing: boolean;
  };
  notes: string[];
  sample_preview: Record<string, string>[];
  uses_full_dataset: boolean;
  error?: string;
};

export type TrainingRecommendations = {
  config: {
    method: string;
    quant: string;
    batch_size: number;
    gradient_accumulation_steps: number;
    max_seq_length: number;
    learning_rate: number;
    epochs: number;
    lora_r: number;
    lora_alpha: number;
    gradient_checkpointing: boolean;
    use_triton: boolean;
    use_fused_ce: boolean;
    train_on_responses_only: boolean;
    use_rslora: boolean;
    packing: boolean;
    dataset_format: string;
    preprocess_dataset?: boolean;
    deduplicate_dataset?: boolean;
    max_eval_samples?: number;
    early_stopping?: boolean;
    early_stopping_patience?: number;
  };
  warnings: string[];
  notes: string[];
  trainable: boolean;
  model_params?: string | null;
  est_training_vram_gb?: number | null;
  hardware_tier?: string;
  dataset_analysis?: DatasetAnalysis;
};

export type CloudGpuCredential = {
  id: string;
  name: string;
  provider_type: string;
  created_at: string;
  config: {
    provider?: string;
    auth_kind?: string;
    region?: string;
    project?: string;
    api_key_configured?: boolean;
    access_key_id_configured?: boolean;
    secret_access_key_configured?: boolean;
    session_token_configured?: boolean;
    ssh_private_key_configured?: boolean;
    bootstrap_command_configured?: boolean;
  };
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
  stages: string[];
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

export type DistillRLJob = {
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

export type DistillRLPreset = {
  id: string;
  label: string;
  stages: string[];
};
