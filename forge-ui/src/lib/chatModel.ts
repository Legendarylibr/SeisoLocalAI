import { api, HardwareProfile, InferenceModelOption, LocalModel } from "@/lib/api";
import { ModelProgressState, initialDownloadProgress, progressFromDownloadEvent, progressFromPreloadEvent } from "@/lib/modelProgress";

export type ChatNavTarget = { modelId?: string | null; repo?: string | null; downloadBytes?: number | null };

export type ModelProgressHandler = (progress: ModelProgressState | null) => void;

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

export function repoFromSource(source: string | null | undefined): string | null {
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

export function pickInferenceModel(list: InferenceModelOption[], target: ChatNavTarget): string {
  return (
    (target.modelId && list.find((m) => m.id === target.modelId)?.id) ||
    (target.repo && list.find((m) => m.source === `hf:${target.repo}`)?.id) ||
    (target.repo &&
      list.find((m) => m.name.toLowerCase().includes(target.repo!.split("/").pop()?.toLowerCase() || ""))?.id) ||
    list.find((m) => m.hardware_fit === "ideal" || m.hardware_fit === "good")?.id ||
    (list.length ? list[0].id : "")
  );
}

function inventoryHasTarget(list: InferenceModelOption[], target: ChatNavTarget): boolean {
  if (target.modelId && list.some((m) => m.id === target.modelId)) return true;
  if (target.repo && list.some((m) => m.source === `hf:${target.repo}`)) return true;
  return false;
}

export function needsHubDownload(list: InferenceModelOption[], target: ChatNavTarget): boolean {
  return !!(target.repo && !inventoryHasTarget(list, target));
}

function streamDownload(
  repo: string,
  variant: "gguf" | "safetensors",
  onProgress?: ModelProgressHandler,
): Promise<string> {
  return new Promise((resolve, reject) => {
    const { promise } = api.streamDownloadModel(
      repo,
      {
        onProgress: (data) => onProgress?.(progressFromDownloadEvent(data)),
        onComplete: (data) => {
          const modelId = String(data.model_id || "");
          if (!modelId) {
            onProgress?.(null);
            reject(new Error("Download completed without model id"));
            return;
          }
          resolve(modelId);
        },
        onError: (msg) => {
          onProgress?.(null);
          reject(new Error(msg));
        },
      },
      variant,
    );
    promise.catch((err) => {
      onProgress?.(null);
      if (!(err instanceof DOMException && err.name === "AbortError")) reject(err);
    });
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
): Promise<string> {
  const initial = await api.listInferenceModels();
  const existing = initial.models.find((m) => m.source === `hf:${repo}`);
  if (existing) return existing.id;
  onProgress?.(initialDownloadProgress(repo, downloadBytes));
  const modelId = await streamDownload(repo, "gguf", onProgress);
  if (!modelId) throw new Error("Download completed without model id");
  return modelId;
}

export function preloadWithProgress(
  modelId: string,
  backend: string,
  onProgress?: ModelProgressHandler,
): Promise<void> {
  return new Promise((resolve, reject) => {
    const { promise } = api.streamPreloadModel(
      modelId,
      backend,
      {
        onProgress: (data) => onProgress?.(progressFromPreloadEvent(data)),
        onComplete: () => resolve(),
        onError: (msg) => {
          onProgress?.(null);
          reject(new Error(msg));
        },
      },
    );
    promise.catch((err) => {
      onProgress?.(null);
      if (!(err instanceof DOMException && err.name === "AbortError")) reject(err);
    });
  });
}

export async function bootstrapChatModels(
  target: ChatNavTarget,
  options: {
    preload?: boolean;
    providerActive?: boolean;
    onProgress?: ModelProgressHandler;
    initialModels?: InferenceModelOption[];
    hwProfile?: HardwareProfile | null;
  } = {},
): Promise<{
  models: InferenceModelOption[];
  selectedId: string;
  backend: string;
  selected: InferenceModelOption | null;
}> {
  let models = options.initialModels ?? (await fetchInferenceModels());
  let downloadedId: string | undefined;

  if (target.repo && !inventoryHasTarget(models, target)) {
    options.onProgress?.(initialDownloadProgress(target.repo, target.downloadBytes ?? undefined));
    downloadedId = await streamDownload(target.repo, "gguf", options.onProgress);
    models = await fetchInferenceModels();
  }

  const selectedId =
    (downloadedId && models.find((m) => m.id === downloadedId)?.id) ||
    pickInferenceModel(models, target);
  const selected = models.find((m) => m.id === selectedId) ?? null;
  const backend = resolveInferenceBackend(selected, options.hwProfile ?? null);

  if (options.preload !== false && selectedId && selected && !options.providerActive) {
    await preloadWithProgress(selectedId, backend, options.onProgress);
  }

  return { models, selectedId, backend, selected };
}
