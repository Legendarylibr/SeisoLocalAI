import { request } from "./client";
import { cachedGet } from "./getCache";
import type { CatalogDataset, TrainableModel, TrainingJob, TrainingMetricsPayload } from "./types";

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
};
