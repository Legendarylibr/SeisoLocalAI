import { api, HardwareProfile, InferenceModelOption, LocalModel } from "@/lib/api";
import { bindAbort, throwIfAborted } from "@/lib/abort";
import { inventoryHasRepo, inventoryMatchesRepo, streamHubModelDownload, ModelProgressHandler } from "@/lib/hubDownload";
import { readStoredModel, writeStoredModel } from "@/lib/modelSelection";
import { progressFromPreloadEvent } from "@/lib/modelProgress";

export const CHAT_MODEL_STORAGE_KEY = "chat";
export const CHAT_BACKEND_STORAGE_KEY = "chat-backend";

type MemoryFitModel = {
  memory_load_blocked?: boolean;
  memory_load_blocked_reason?: string | null;
  est_vram_mb?: number;
  hardware_note?: string;
};

/** True when a model's estimated runtime memory exceeds free VRAM/RAM headroom. */
export function modelMemoryBlocked(
  model: MemoryFitModel | null | undefined,
  headroomMb?: number,
): boolean {
  if (!model) return false;
  if (model.memory_load_blocked) return true;
  if (headroomMb && model.est_vram_mb) return model.est_vram_mb > headroomMb;
  return false;
}

export function modelMemoryBlockReason(model: MemoryFitModel | null | undefined): string {
  return (
    model?.memory_load_blocked_reason ||
    model?.hardware_note ||
    "This model exceeds available memory on your machine."
  );
}

type ChatNavTarget = { modelId?: string | null; repo?: string | null; downloadBytes?: number | null };

type BootstrapOptions = {
  preload?: boolean;
  providerActive?: boolean;
  onProgress?: ModelProgressHandler;
  initialModels?: InferenceModelOption[];
  hwProfile?: HardwareProfile | null;
  signal?: AbortSignal;
};

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

/** Resolve inference backend — trusts server default_backend, then hardware preference. */
export function resolveInferenceBackend(
  model: InferenceModelOption | null,
  hwProfile: HardwareProfile | null,
  override?: string,
): string {
  if (!model) return "";
  const available = model.backends?.length ? model.backends : model.default_backend ? [model.default_backend] : [];
  if (available.length === 0) return "";
  if (available.length === 1) return available[0];
  if (override && override !== "auto" && available.includes(override)) return override;

  if (model.default_backend && available.includes(model.default_backend)) {
    return model.default_backend;
  }

  const preferred = hwProfile?.preferred_inference_backend;
  if (preferred && available.includes(preferred)) return preferred;

  return available[0];
}

export function pickInferenceModel(
  list: InferenceModelOption[],
  target: ChatNavTarget = {},
  storedId?: string | null,
): string {
  return (
    (target.modelId && list.find((m) => m.id === target.modelId)?.id) ||
    (target.repo && list.find((m) => inventoryMatchesRepo(m, target.repo!))?.id) ||
    (storedId && list.find((m) => m.id === storedId)?.id) ||
    list.find((m) => !modelMemoryBlocked(m) && (m.hardware_fit === "ideal" || m.hardware_fit === "good"))?.id ||
    list.find((m) => !modelMemoryBlocked(m))?.id ||
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

async function downloadChatModel(
  repo: string,
  onProgress?: ModelProgressHandler,
  options: { signal?: AbortSignal; downloadBytes?: number | null } = {},
): Promise<string> {
  throwIfAborted(options.signal);
  // Chat always downloads GGUF for llama.cpp inference — never safetensors snapshots.
  return streamHubModelDownload(repo, "gguf", onProgress, {
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
  const existing = initial.models.find((m) => inventoryMatchesRepo(m, repo));
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

/** Load chat inventory, optionally download a Hub model, pick defaults, and preload. */
export async function bootstrapChatSession(
  target: ChatNavTarget = {},
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

  const useStoredPreferences = !hasChatNavTarget(target);
  const selectedId =
    downloadedId ||
    pickInferenceModel(models, target, readStoredModel(CHAT_MODEL_STORAGE_KEY));
  const storedBackend = readStoredModel(CHAT_BACKEND_STORAGE_KEY);
  const selected = models.find((m) => m.id === selectedId) ?? null;
  const backend = resolveInferenceBackend(
    selected,
    options.hwProfile ?? null,
    useStoredPreferences ? storedBackend ?? undefined : undefined,
  );

  if (target.repo && !selected) {
    throw new Error(
      `Download finished but ${target.repo} was not found in local inventory. Try again or check Settings → Hugging Face.`,
    );
  }
  if (selected && !backend) {
    const fmt = selected.format?.toLowerCase();
    const hint =
      fmt === "gguf"
        ? "This GGUF is not available for local chat. Install or update llama.cpp, or choose a GGUF architecture supported by your llama.cpp runtime."
        : "No installed local inference engine can load this model. Install MLX or PyTorch support.";
    throw new Error(hint);
  }

  const headroomMb = options.hwProfile?.vram_headroom_mb;
  if (selected && modelMemoryBlocked(selected, headroomMb)) {
    throw new Error(modelMemoryBlockReason(selected));
  }

  let loadedBackend = backend;
  if (options.preload !== false && selectedId && selected && !options.providerActive) {
    loadedBackend = await preloadWithProgress(selectedId, backend, options.onProgress, options.signal);
    writeStoredModel(CHAT_MODEL_STORAGE_KEY, selectedId);
    writeStoredModel(CHAT_BACKEND_STORAGE_KEY, loadedBackend);
  }

  return { models, selectedId, backend: loadedBackend, selected };
}

/** Open /chat without Hub params — restores last model from localStorage. */
export async function initializeChatSession(
  options: BootstrapOptions = {},
): Promise<{
  models: InferenceModelOption[];
  selectedId: string;
  backend: string;
  selected: InferenceModelOption | null;
}> {
  return bootstrapChatSession({}, options);
}
