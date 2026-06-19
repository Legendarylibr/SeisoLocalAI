import { API, request } from "./client";
import type { ExportJob, HubPublishFields, PublishableModel } from "./types";

export const exportApi = {
  listPublishableOutputs: () => request<PublishableModel[]>("/export/publishable"),
  listExportProfiles: () =>
    request<{ id: string; formats: string[]; default_gguf_quants: string[] }[]>("/export/profiles"),
  precheckHubExport: (body: {
    hub: HubPublishFields;
    formats?: string[];
    profile?: string;
    gguf_quantizations?: string[];
  }) =>
    request<{
      ok: boolean;
      repo_id: string;
      errors: string[];
      warnings: string[];
      model_card_preview: string;
    }>("/export/precheck", { method: "POST", body: JSON.stringify(body) }),
  listExportJobs: () => request<ExportJob[]>("/export/jobs"),
  startExport: (
    checkpoint: string,
    formats: string[],
    hub?: HubPublishFields,
    rlQuantJobId?: string,
    profile?: string,
    ggufQuantizations?: string[],
  ) =>
    request<{ job_id: string }>("/export/jobs", {
      method: "POST",
      body: JSON.stringify({
        checkpoint,
        formats,
        profile: profile || null,
        gguf_quantizations: ggufQuantizations?.length ? ggufQuantizations : null,
        hub: hub || null,
        rl_quant_job_id: rlQuantJobId || null,
      }),
    }),
  publishToHub: (body: {
    model_id?: string;
    export_job_id?: string;
    output_path?: string;
    hub: HubPublishFields;
  }) =>
    request<{ repo_id: string; path: string; log: string }>("/export/publish", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  downloadExportOutput: (jobId: string, key = "gguf") =>
    fetch(`${API}/export/outputs/${jobId}/download?key=${encodeURIComponent(key)}`, {
      credentials: "include",
    }),
};
