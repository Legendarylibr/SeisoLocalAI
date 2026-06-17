import { api, TrainableModel } from "@/lib/api";
import { inventoryHasRepo, streamHubModelDownload, ModelProgressHandler } from "@/lib/hubDownload";

export async function fetchTrainableModels(): Promise<TrainableModel[]> {
  return (await api.listTrainingModels()).models;
}

export function isTrainModelCached(models: TrainableModel[], repo: string): boolean {
  return models.some((m) => m.repo_id === repo);
}

/** Download safetensors snapshot when missing from local training inventory. */
export async function ensureTrainHubModel(
  repo: string,
  onProgress?: ModelProgressHandler,
  downloadBytes?: number,
  signal?: AbortSignal,
): Promise<void> {
  const models = await fetchTrainableModels();
  if (isTrainModelCached(models, repo) || inventoryHasRepo(models, repo)) {
    return;
  }
  await streamHubModelDownload(repo, "safetensors", onProgress, { signal, downloadBytes });
}
