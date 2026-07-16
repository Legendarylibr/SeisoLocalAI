import { invalidateApiCache } from "@/lib/api/getCache";
import { api } from "@/lib/api";
import { throwIfAborted } from "@/lib/abort";
import { ModelProgressState, formatBytes, initialDownloadProgress, progressFromDownloadEvent } from "@/lib/modelProgress";

export type ModelProgressHandler = (progress: ModelProgressState | null) => void;

/** Stream a Hugging Face repo into local inventory with live byte progress. */
export function streamHubModelDownload(
  repo: string,
  variant: "gguf" | "safetensors",
  onProgress?: ModelProgressHandler,
  options: { signal?: AbortSignal; downloadBytes?: number; filename?: string } = {},
): Promise<string> {
  const { signal, downloadBytes, filename } = options;
  throwIfAborted(signal);
  onProgress?.(initialDownloadProgress(repo, downloadBytes));

  return new Promise((resolve, reject) => {
    let settled = false;
    const finishResolve = (value: string) => {
      if (settled) return;
      settled = true;
      resolve(value);
    };
    const finishReject = (err: Error) => {
      if (settled) return;
      settled = true;
      reject(err);
    };

    const { promise, abort } = api.streamDownloadModel(
      repo,
      {
        onProgress: (data) => {
          const progress = progressFromDownloadEvent(data);
          const total = typeof data.total_bytes === "number" ? data.total_bytes : 0;
          if (downloadBytes && total > downloadBytes * 1.25) {
            progress.label = `Resolved actual download size: ${formatBytes(total)} · ${repo}`;
          }
          onProgress?.(progress);
        },
        onComplete: (data) => {
          const modelId = String(data.model_id || "");
          if (!modelId) {
            onProgress?.(null);
            finishReject(new Error("Download completed without model id"));
            return;
          }
          invalidateApiCache("/inference/models");
          invalidateApiCache("/training/models");
          invalidateApiCache("/models");
          if (modelId) invalidateApiCache(`/inference/models/${modelId}/variants`);
          finishResolve(modelId);
        },
        onError: (msg) => {
          onProgress?.(null);
          finishReject(new Error(msg));
        },
      },
      variant,
      filename ? { filename } : {},
    );

    const onAbort = () => {
      abort();
      onProgress?.(null);
      finishReject(new DOMException("Aborted", "AbortError"));
    };

    if (signal) {
      if (signal.aborted) {
        onAbort();
        return;
      }
      signal.addEventListener("abort", onAbort, { once: true });
    }

    promise
      .then(async () => {
        if (!settled) {
          invalidateApiCache("/inference/models");
          invalidateApiCache("/training/models");
          invalidateApiCache("/models");
          const recovered = findInventoryModelId((await api.listInferenceModels()).models, repo);
          if (recovered) {
            finishResolve(recovered);
            return;
          }
          onProgress?.(null);
          finishReject(new Error("Download stream ended before completion and the model was not found in inventory"));
        }
      })
      .catch((err) => {
        onProgress?.(null);
        if (settled) return;
        if (err instanceof DOMException && err.name === "AbortError") {
          finishReject(err);
          return;
        }
        finishReject(err instanceof Error ? err : new Error(String(err)));
      })
      .finally(() => {
        signal?.removeEventListener("abort", onAbort);
      });
  });
}

export function trainPath(repo: string, downloadBytes?: number | null): string {
  const params = new URLSearchParams({ model: repo, download: "1" });
  if (downloadBytes && downloadBytes > 0) {
    params.set("bytes", String(downloadBytes));
  }
  return `/train?${params.toString()}`;
}

export function inventoryMatchesRepo(
  model: { source?: string | null; metadata?: Record<string, unknown> | null },
  repo: string,
): boolean {
  const source = model.source || "";
  // Canonical hf:org/model plus multi-quant siblings hf:org/model:file.gguf
  if (source === `hf:${repo}` || source.startsWith(`hf:${repo}:`)) return true;
  const metaRepo = typeof model.metadata?.repo_id === "string" ? model.metadata.repo_id : null;
  return metaRepo === repo;
}

export function inventoryHasRepo(
  list: Array<{ source?: string | null; metadata?: Record<string, unknown> | null }>,
  repo: string,
): boolean {
  return list.some((m) => inventoryMatchesRepo(m, repo));
}

export function findInventoryModelId(
  list: Array<{ id: string; source?: string | null; metadata?: Record<string, unknown> | null }>,
  repo: string,
): string | undefined {
  return list.find((m) => inventoryMatchesRepo(m, repo))?.id;
}
