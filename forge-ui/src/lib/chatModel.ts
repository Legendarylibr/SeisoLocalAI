import { api, HardwareProfile, InferenceModelOption, LocalModel } from "@/lib/api";
import { ROUTER_MODEL_ID } from "@/lib/api/types";
import { bindAbort, throwIfAborted } from "@/lib/abort";
import { inventoryHasRepo, inventoryMatchesRepo, streamHubModelDownload, ModelProgressHandler } from "@/lib/hubDownload";
import { readStoredModel, writeStoredModel } from "@/lib/modelSelection";
import { progressFromPreloadEvent } from "@/lib/modelProgress";
import { readStoredContextWindowForModel } from "@/lib/chatContext";

export const CHAT_MODEL_STORAGE_KEY = "chat";
export const CHAT_BACKEND_STORAGE_KEY = "chat-backend";

type MemoryFitModel = {
  memory_load_blocked?: boolean;
  memory_load_blocked_reason?: string | null;
  est_vram_mb?: number;
  hardware_note?: string;
  hardware_fit?: string;
};

type SelectableModel = MemoryFitModel & {
  selectable?: boolean;
  status?: string;
};

/** True when a model cannot be loaded for chat because the artifact is unavailable. */
export function modelMemoryBlocked(
  model: SelectableModel | null | undefined,
  _headroomMb?: number,
): boolean {
  if (!model) return false;
  if (model.memory_load_blocked) return true;
  if (model.selectable === false || model.status === "incomplete") return true;
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
  maxTokens?: number;
  nCtx?: number | null;
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

function isSelectableChatModel(model: InferenceModelOption | undefined): boolean {
  if (!model) return false;
  if (model.selectable === false || model.status === "incomplete") return false;
  return true;
}

export function pickInferenceModel(
  list: InferenceModelOption[],
  target: ChatNavTarget = {},
  storedId?: string | null,
): string {
  const ready = list.filter(isSelectableChatModel);
  return (
    (target.modelId && ready.find((m) => m.id === target.modelId)?.id) ||
    (target.repo && ready.find((m) => inventoryMatchesRepo(m, target.repo!))?.id) ||
    (storedId && ready.find((m) => m.id === storedId)?.id) ||
    ready.find((m) => !modelMemoryBlocked(m) && (m.hardware_fit === "ideal" || m.hardware_fit === "good"))?.id ||
    ready.find((m) => !modelMemoryBlocked(m))?.id ||
    (ready.length ? ready[0].id : "")
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

export type PreloadInferenceOptions = {
  maxTokens?: number;
  nCtx?: number | null;
};

export function preloadWithProgress(
  modelId: string,
  backend: string,
  onProgress?: ModelProgressHandler,
  signal?: AbortSignal,
  inference?: PreloadInferenceOptions,
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
      {
        max_tokens: inference?.maxTokens,
        n_ctx: inference?.nCtx,
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
  kind?: InferenceModelOption["kind"],
): boolean {
  if (modelId === ROUTER_MODEL_ID || kind === "router" || backend === "router") {
    return !!modelId;
  }
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
    const installHint = selected.install_hints?.[0];
    const hint =
      fmt === "gguf"
        ? installHint
          ? `GGUF chat needs llama.cpp. Run: ${installHint}`
          : "This GGUF is not available for local chat. Install llama-cpp-python (pip install \"llama-cpp-python>=0.3\") or re-run start."
        : "No installed local inference engine can load this model. Install MLX or PyTorch support.";
    throw new Error(hint);
  }

  const headroomMb = options.hwProfile?.vram_headroom_mb;
  if (selected && modelMemoryBlocked(selected, headroomMb)) {
    throw new Error(modelMemoryBlockReason(selected));
  }

  let loadedBackend = backend;
  if (
    options.preload !== false &&
    selectedId &&
    selected &&
    selected.kind !== "router" &&
    !options.providerActive
  ) {
    let nCtx = options.nCtx;
    if (nCtx === undefined) {
      const ceiling =
        typeof selected.context_ceiling === "number" ? selected.context_ceiling : 131072;
      const stored = readStoredContextWindowForModel(selectedId, ceiling);
      nCtx = stored === "auto" ? null : stored;
    }
    loadedBackend = await preloadWithProgress(
      selectedId,
      backend,
      options.onProgress,
      options.signal,
      { maxTokens: options.maxTokens, nCtx },
    );
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
