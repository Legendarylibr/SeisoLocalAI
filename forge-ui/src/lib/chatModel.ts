import { api, InferenceModelOption, LocalModel } from "@/lib/api";
import { ModelProgressState, initialDownloadProgress, progressFromDownloadEvent, progressFromPreloadEvent } from "@/lib/modelProgress";

export type ChatNavTarget = { modelId?: string | null; repo?: string | null };

export type ModelProgressHandler = (progress: ModelProgressState | null) => void;

export function chatPath(target: ChatNavTarget = {}): string {
  const params = new URLSearchParams();
  if (target.modelId) params.set("model", target.modelId);
  if (target.repo) params.set("repo", target.repo);
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
          resolve(String(data.model_id || ""));
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

/** Ensure a catalog repo is in inventory; download GGUF mirror when missing. */
export async function ensureHubChatModel(
  repo: string,
  onProgress?: ModelProgressHandler,
): Promise<string> {
  const initial = await api.listInferenceModels();
  const existing = initial.models.find((m) => m.source === `hf:${repo}`);
  if (existing) return existing.id;
  onProgress?.(initialDownloadProgress(repo));
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
        onComplete: () => {
          onProgress?.(null);
          resolve();
        },
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
  } = {},
): Promise<{
  models: InferenceModelOption[];
  selectedId: string;
  backend: string;
  selected: InferenceModelOption | null;
}> {
  let models = options.initialModels ?? (await api.listInferenceModels()).models;

  if (target.repo && !inventoryHasTarget(models, target)) {
    options.onProgress?.(initialDownloadProgress(target.repo));
    await streamDownload(target.repo, "gguf", options.onProgress);
    models = (await api.listInferenceModels()).models;
  }

  const selectedId = pickInferenceModel(models, target);
  const selected = models.find((m) => m.id === selectedId) ?? null;
  const backend = selected?.default_backend || "auto";

  if (options.preload !== false && selectedId && selected && !options.providerActive) {
    await preloadWithProgress(selectedId, backend, options.onProgress);
  }

  return { models, selectedId, backend, selected };
}
