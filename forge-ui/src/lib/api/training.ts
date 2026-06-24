import { request } from "./client";
import { cachedGet } from "./getCache";
import type { CatalogDataset, TrainableModel, TrainingJob, TrainingMetricsPayload, TrainingRecommendations } from "./types";

export const trainingApi = {
  startTraining: (
    config: Record<string, unknown>,
    multi_gpu = false,
    export_on_complete?: {
      profile?: string;
      formats?: string[];
      gguf_quantizations?: string[];
    },
  ) =>
    request<{ job_id: string; status: string }>("/training/jobs", {
      method: "POST",
      body: JSON.stringify({ config, multi_gpu, export_on_complete }),
    }),
  listTrainingJobs: () => request<TrainingJob[]>("/training/jobs"),
  listTrainingModels: () =>
    cachedGet<{ models: TrainableModel[]; total: number }>("/training/models", 120_000),
  searchDatasets: (q: string, limit = 12) => {
    const params = new URLSearchParams({ q, limit: String(limit) });
    return request<{ datasets: CatalogDataset[]; total: number }>(`/training/datasets?${params}`);
  },
  getTrainingMetrics: (jobId: string) => request<TrainingMetricsPayload>(`/training/jobs/${jobId}/metrics`),
  getTrainingRecommendations: (modelId: string, dataset: string) => {
    const params = new URLSearchParams();
    if (modelId) params.set("model_id", modelId);
    if (dataset) params.set("dataset", dataset);
    return request<TrainingRecommendations>(`/training/recommendations?${params}`);
  },
  validateDataset: (dataset: string, datasetFormat: string = "auto") =>
    request<{ valid: boolean; kept?: number; resolved_format?: string; error?: string }>(
      "/training/validate-dataset",
      {
        method: "POST",
        body: JSON.stringify({ dataset, dataset_format: datasetFormat }),
      },
    ),
};
