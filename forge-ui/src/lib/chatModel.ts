import { api, HardwareProfile, InferenceModelOption, LocalModel } from "@/lib/api";
import { inventoryHasRepo, streamHubModelDownload, ModelProgressHandler } from "@/lib/hubDownload";
import { readStoredModel, writeStoredModel } from "@/lib/modelSelection";
import { progressFromPreloadEvent } from "@/lib/modelProgress";

export const CHAT_MODEL_STORAGE_KEY = "chat";
export const CHAT_BACKEND_STORAGE_KEY = "chat-backend";

type ChatNavTarget = { modelId?: string | null; repo?: string | null; downloadBytes?: number | null };

type BootstrapOptions = {
  preload?: boolean;
  providerActive?: boolean;
  onProgress?: ModelProgressHandler;
  initialModels?: InferenceModelOption[];
  hwProfile?: HardwareProfile | null;
  signal?: AbortSignal;
};

function throwIfAborted(signal?: AbortSignal) {
  if (signal?.aborted) {
    throw new DOMException("Aborted", "AbortError");
  }
}

function bindAbort<T>(
  signal: AbortSignal | undefined,
  abort: () => void,
  promise: Promise<T>,
): Promise<T> {
  if (!signal) return promise;
  if (signal.aborted) {
    abort();
    return Promise.reject(new DOMException("Aborted", "AbortError"));
  }
  const onAbort = () => abort();
  signal.addEventListener("abort", onAbort, { once: true });
  return promise.finally(() => signal.removeEventListener("abort", onAbort));
}

export function chatPath(target: ChatNavTarget = {}): string {
  const params = new URLSearchParams();
  if (target.modelId) params.set("model", target.modelId);
  if (target.repo) params.set("repo", target.repo);
  if (target.downloadBytes && target.downloadBytes > 0) {
    params.set("bytes", String(target.downloadBytes));
  }
  const qs = params.toString();
  return qs ? `/chat?${qs}` : "/chat";
}

function repoFromSource(source: string | null | undefined): string | null {
  if (!source?.startsWith("hf:")) return null;
  return source.slice(3);
}

export function chatPathForLocalModel(model: LocalModel): string {
  return chatPath({
    modelId: model.id,
    repo: repoFromSource(model.source),
  });
}

/** Pick Ollama vs llama.cpp (or MLX/torch) from model availability and hardware profile. */
export function resolveInferenceBackend(
  model: InferenceModelOption | null,
  hwProfile: HardwareProfile | null,
  override?: string,
): string {
  if (!model) return "llamacpp";
  const available = model.backends?.length ? model.backends : [model.default_backend || "llamacpp"];
  if (available.length === 1) return available[0];
  if (override && override !== "auto" && available.includes(override)) return override;

  const preferred = hwProfile?.preferred_inference_backend;
  if (preferred && available.includes(preferred)) return preferred;

  if (available.includes("ollama") && available.includes("llamacpp")) {
    const tier = hwProfile?.tier;
    if (tier === "cpu_only" || tier === "edge") return "llamacpp";
    const headroom = hwProfile?.vram_headroom_mb ?? 0;
    return headroom >= 8000 ? "ollama" : "llamacpp";
  }

  if (model.default_backend && available.includes(model.default_backend)) {
    return model.default_backend;
  }
  return available[0];
}

export function pickInferenceModel(
  list: InferenceModelOption[],
  target: ChatNavTarget = {},
  storedId?: string | null,
): string {
  return (
    (target.modelId && list.find((m) => m.id === target.modelId)?.id) ||
    (target.repo && list.find((m) => m.source === `hf:${target.repo}`)?.id) ||
    (storedId && list.find((m) => m.id === storedId)?.id) ||
    list.find((m) => m.hardware_fit === "ideal" || m.hardware_fit === "good")?.id ||
    (list.length ? list[0].id : "")
  );
}

export function hasChatNavTarget(target: ChatNavTarget): boolean {
  return !!(target.modelId || target.repo);
}

function inventoryHasTarget(list: InferenceModelOption[], target: ChatNavTarget): boolean {
  if (target.modelId && list.some((m) => m.id === target.modelId)) return true;
  if (target.repo && inventoryHasRepo(list, target.repo)) return true;
  return false;
}

export function needsHubDownload(list: InferenceModelOption[], target: ChatNavTarget): boolean {
  return !!(target.repo && !inventoryHasTarget(list, target));
}

async function resolveChatDownloadVariant(): Promise<"gguf" | "safetensors"> {
  try {
    const status = await api.hfStatus();
    const { llamacpp, mlx, torch } = status.runtime;
    if (!llamacpp && (mlx || torch)) return "safetensors";
    return "gguf";
  } catch {
    return "gguf";
  }
}

async function downloadChatModel(
  repo: string,
  onProgress?: ModelProgressHandler,
  options: { signal?: AbortSignal; downloadBytes?: number | null } = {},
): Promise<string> {
  throwIfAborted(options.signal);
  const variant = await resolveChatDownloadVariant();
  return streamHubModelDownload(repo, variant, onProgress, {
    signal: options.signal,
    downloadBytes: options.downloadBytes ?? undefined,
  });
}

export async function fetchInferenceModels(): Promise<InferenceModelOption[]> {
  return (await api.listInferenceModels()).models;
}

/** Ensure a catalog repo is in inventory; download GGUF mirror when missing. */
export async function ensureHubChatModel(
  repo: string,
  onProgress?: ModelProgressHandler,
  downloadBytes?: number,
  signal?: AbortSignal,
): Promise<string> {
  throwIfAborted(signal);
  const initial = await api.listInferenceModels();
  const existing = initial.models.find((m) => m.source === `hf:${repo}`);
  if (existing) return existing.id;
  return downloadChatModel(repo, onProgress, { signal, downloadBytes });
}

export function preloadWithProgress(
  modelId: string,
  backend: string,
  onProgress?: ModelProgressHandler,
  signal?: AbortSignal,
): Promise<string> {
  throwIfAborted(signal);
  return new Promise((resolve, reject) => {
    const { promise, abort } = api.streamPreloadModel(
      modelId,
      backend,
      {
        onProgress: (data) => onProgress?.(progressFromPreloadEvent(data)),
        onComplete: (data) => {
          const resolvedBackend =
            typeof data.backend === "string" && data.backend ? data.backend : backend;
          resolve(resolvedBackend);
        },
        onError: (msg) => {
          onProgress?.(null);
          reject(new Error(msg));
        },
      },
    );
    bindAbort(signal, abort, promise).catch((err) => {
      onProgress?.(null);
      if (!(err instanceof DOMException && err.name === "AbortError")) reject(err);
    });
  });
}

export function isChatModelReady(
  modelId: string | null,
  backend: string,
  loadedModelId: string | null,
  loadedBackend: string | null,
): boolean {
  return !!(modelId && loadedModelId === modelId && loadedBackend === backend);
}

export async function bootstrapChatModels(
  target: ChatNavTarget,
  options: BootstrapOptions = {},
): Promise<{
  models: InferenceModelOption[];
  selectedId: string;
  backend: string;
  selected: InferenceModelOption | null;
}> {
  throwIfAborted(options.signal);
  let models = options.initialModels ?? (await fetchInferenceModels());
  throwIfAborted(options.signal);
  let downloadedId: string | undefined;

  if (target.repo && !inventoryHasTarget(models, target)) {
    downloadedId = await downloadChatModel(target.repo, options.onProgress, {
      signal: options.signal,
      downloadBytes: target.downloadBytes,
    });
    models = await fetchInferenceModels();
    throwIfAborted(options.signal);
  }

  const selectedId =
    (downloadedId && models.find((m) => m.id === downloadedId)?.id) ||
    pickInferenceModel(models, target, readStoredModel(CHAT_MODEL_STORAGE_KEY));
  const storedBackend = readStoredModel(CHAT_BACKEND_STORAGE_KEY);
  const selected = models.find((m) => m.id === selectedId) ?? null;
  const backend = resolveInferenceBackend(
    selected,
    options.hwProfile ?? null,
    !target.modelId && !target.repo ? storedBackend ?? undefined : undefined,
  );

  let loadedBackend = backend;
  if (options.preload !== false && selectedId && selected && !options.providerActive) {
    loadedBackend = await preloadWithProgress(selectedId, backend, options.onProgress, options.signal);
    writeStoredModel(CHAT_MODEL_STORAGE_KEY, selectedId);
    writeStoredModel(CHAT_BACKEND_STORAGE_KEY, loadedBackend);
  }

  return { models, selectedId, backend: loadedBackend, selected };
}

/** Load chat inventory and pick a default model when opening /chat without Hub params. */
export async function initializeChatSession(
  options: BootstrapOptions = {},
): Promise<{
  models: InferenceModelOption[];
  selectedId: string;
  backend: string;
  selected: InferenceModelOption | null;
}> {
  throwIfAborted(options.signal);
  const models = options.initialModels ?? (await fetchInferenceModels());
  throwIfAborted(options.signal);
  const storedId = readStoredModel(CHAT_MODEL_STORAGE_KEY);
  const storedBackend = readStoredModel(CHAT_BACKEND_STORAGE_KEY);
  const selectedId = pickInferenceModel(models, {}, storedId);
  const selected = models.find((m) => m.id === selectedId) ?? null;
  const backend = resolveInferenceBackend(
    selected,
    options.hwProfile ?? null,
    storedBackend ?? undefined,
  );

  let loadedBackend = backend;
  if (options.preload !== false && selectedId && selected && !options.providerActive) {
    loadedBackend = await preloadWithProgress(selectedId, backend, options.onProgress, options.signal);
    writeStoredModel(CHAT_MODEL_STORAGE_KEY, selectedId);
    writeStoredModel(CHAT_BACKEND_STORAGE_KEY, loadedBackend);
  }

  return { models, selectedId, backend: loadedBackend, selected };
}
