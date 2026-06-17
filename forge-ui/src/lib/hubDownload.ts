import { api } from "@/lib/api";
import { ModelProgressState, initialDownloadProgress, progressFromDownloadEvent } from "@/lib/modelProgress";

export type ModelProgressHandler = (progress: ModelProgressState | null) => void;

function throwIfAborted(signal?: AbortSignal) {
  if (signal?.aborted) {
    throw new DOMException("Aborted", "AbortError");
  }
}

/** Stream a Hugging Face repo into local inventory with live byte progress. */
export function streamHubModelDownload(
  repo: string,
  variant: "gguf" | "safetensors",
  onProgress?: ModelProgressHandler,
  options: { signal?: AbortSignal; downloadBytes?: number } = {},
): Promise<string> {
  const { signal, downloadBytes } = options;
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
        onProgress: (data) => onProgress?.(progressFromDownloadEvent(data)),
        onComplete: (data) => {
          const modelId = String(data.model_id || "");
          if (!modelId) {
            onProgress?.(null);
            finishReject(new Error("Download completed without model id"));
            return;
          }
          finishResolve(modelId);
        },
        onError: (msg) => {
          onProgress?.(null);
          finishReject(new Error(msg));
        },
      },
      variant,
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

export function inventoryHasRepo(
  list: Array<{ source?: string | null }>,
  repo: string,
): boolean {
  const source = `hf:${repo}`;
  return list.some((m) => m.source === source);
}
