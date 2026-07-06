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
};

export function useStagePipelinePage<TJob extends { id: string }>({
  fallbackStages,
  loadPresets,
  listJobs,
  startJob,
  streamPath,
}: StagePipelinePageOptions<TJob>) {
  const [jobs, setJobs] = useState<TJob[]>([]);
  const [starting, setStarting] = useState(false);
  const { models: localModels, loading: modelsLoading } = useTrainingModels();

  const presets = useStagePipelinePresets(fallbackStages, loadPresets);
  const { logs, result, activeJob, resetStream, watchJob } = usePipelineJobStream();

  useEffect(() => {
    listJobs().then(setJobs).catch(console.error);
  }, [listJobs]);

  const refreshJobs = useCallback(() => {
    listJobs().then(setJobs).catch(console.error);
  }, [listJobs]);

  const runPipeline = useCallback(
    async (body: Record<string, unknown>) => {
      setStarting(true);
      resetStream();
      try {
        const res = await startJob(body);
        watchJob(streamPath(res.job_id), res.job_id, {
          onResult: () => {
            invalidateApiCache("/inference/models");
            invalidateApiCache("/training/models");
            refreshJobs();
          },
        });
        refreshJobs();
      } catch (err) {
        const message = err instanceof Error ? err.message : "Failed to start pipeline";
        console.error(err);
        throw new Error(message);
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
    runPipeline,
    logs,
    result,
    activeJob,
    ...presets,
  };
}
