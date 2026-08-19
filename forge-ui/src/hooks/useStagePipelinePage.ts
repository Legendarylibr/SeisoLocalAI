import { useCallback, useEffect, useState } from "react";
import { useTrainingModels } from "@/context/TrainingModelsContext";
import { invalidateApiCache } from "@/lib/api/getCache";
import { usePipelineJobStream } from "@/hooks/usePipelineJobStream";
import { useStagePipelinePresets } from "@/hooks/useStagePipelinePresets";

type StagePreset = { id: string; label: string; stages: string[] };

type PresetsResponse = {
  presets: StagePreset[];
  stages: string[];
  help: Record<string, string>;
  defaults?: Record<string, string>;
};

type StagePipelinePageOptions<TJob> = {
  fallbackStages: string[];
  loadPresets: () => Promise<PresetsResponse>;
  listJobs: () => Promise<TJob[]>;
  startJob: (body: Record<string, unknown>) => Promise<{ job_id: string }>;
  streamPath: (jobId: string) => string;
  initialPreset?: string;
};

export function useStagePipelinePage<TJob extends { id: string }>({
  fallbackStages,
  loadPresets,
  listJobs,
  startJob,
  streamPath,
  initialPreset = "smoke",
}: StagePipelinePageOptions<TJob>) {
  const [jobs, setJobs] = useState<TJob[]>([]);
  const [starting, setStarting] = useState(false);
  const [startError, setStartError] = useState<string | null>(null);
  const { models: localModels, loading: modelsLoading } = useTrainingModels();

  const presets = useStagePipelinePresets(fallbackStages, loadPresets, initialPreset);
  const { logs, result, activeJob, resetStream, watchJob } = usePipelineJobStream();

  useEffect(() => {
    listJobs().then(setJobs).catch(console.error);
  }, [listJobs]);

  const refreshJobs = useCallback(() => {
    listJobs().then(setJobs).catch(console.error);
  }, [listJobs]);

  const runPipeline = useCallback(
    async (
      body: Record<string, unknown>,
      opts?: { onEvent?: (event: string, data: string) => void },
    ) => {
      setStarting(true);
      setStartError(null);
      resetStream();
      try {
        const res = await startJob(body);
        const refreshAfterJob = () => {
          invalidateApiCache("/inference/models");
          invalidateApiCache("/training/models");
          refreshJobs();
        };
        watchJob(streamPath(res.job_id), res.job_id, {
          onEvent: opts?.onEvent,
          onResult: refreshAfterJob,
          onError: refreshAfterJob,
          onStreamError: refreshAfterJob,
        });
        refreshJobs();
      } catch (err) {
        const message = err instanceof Error ? err.message : "Failed to start pipeline";
        console.error(err);
        setStartError(message);
      } finally {
        setStarting(false);
      }
    },
    [startJob, streamPath, refreshJobs, resetStream, watchJob],
  );

  return {
    jobs,
    localModels,
    modelsReady: !modelsLoading,
    starting,
    startError,
    runPipeline,
    logs,
    result,
    activeJob,
    ...presets,
  };
}
