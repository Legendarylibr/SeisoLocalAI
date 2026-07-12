import { useCallback, useEffect, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import {
  api,
  CloudGpuCredential,
  DatasetAnalysis,
  subscribeSSE,
  SystemMetrics,
  TrainingJob,
  TrainingMetricPoint,
  TrainingRecommendations,
} from "@/lib/api";
import { invalidateApiCache } from "@/lib/api/getCache";
import { appendBoundedLog } from "@/lib/api/sse";
import { initialDownloadProgress, ModelProgressState } from "@/lib/modelProgress";
import { ensureTrainHubModel, isTrainModelCached } from "@/lib/trainModel";

import { useTrainingModels } from "@/context/TrainingModelsContext";
import { readStoredModel, writeStoredModel } from "@/lib/modelSelection";
import { HfBaseModelPicker } from "@/components/HfBaseModelPicker";
import { HfDatasetPicker } from "@/components/HfDatasetPicker";
import { ModelLoadProgress } from "@/components/ModelLoadProgress";
import { FormSection } from "@/components/research/FormSection";
import { DataTable } from "@/components/research/DataTable";
import { LogStream } from "@/components/research/LogStream";
import { StudioPageShell } from "@/components/StudioPageShell";
import { StudioCardBody } from "@/components/studio/StudioCardBody";
import { StudioCardHeader } from "@/components/studio/StudioCardHeader";
import { Tabs } from "@/components/Tabs";
import { TrainingMetricsDashboard } from "@/components/TrainingMetricsDashboard";
import { useHardwareProfile } from "@/hooks/useHardware";

type TrainStudioTab = "setup" | "distributed" | "cloud";

export function TrainPage() {
  const { profile: hw } = useHardwareProfile();
  const [searchParams, setSearchParams] = useSearchParams();
  const pendingModel = searchParams.get("model");
  const pendingDownload = searchParams.get("download") === "1";
  const pendingDownloadBytes = (() => {
    const raw = searchParams.get("bytes");
    if (!raw) return undefined;
    const n = Number(raw);
    return Number.isFinite(n) && n > 0 ? n : undefined;
  })();
  const [jobs, setJobs] = useState<TrainingJob[]>([]);
  const { models: localModels, refresh: refreshLocalModels } = useTrainingModels();
  const [modelId, setModelId] = useState("");
  // Empty until the user picks a path/hub id — avoids full-corpus analysis on tab open.
  const [dataset, setDataset] = useState("");
  const [method, setMethod] = useState("lora");
  const [quant, setQuant] = useState("4bit");
  const [datasetFormat, setDatasetFormat] = useState("auto");
  const [epochs, setEpochs] = useState(5);
  const [earlyStopping, setEarlyStopping] = useState(true);
  const [earlyStoppingPatience, setEarlyStoppingPatience] = useState(3);
  const [preprocessDataset, setPreprocessDataset] = useState(true);
  const [batchSize, setBatchSize] = useState(2);
  const [lr, setLr] = useState(0.0002);
  const [maxSeq, setMaxSeq] = useState(2048);
  const [loraR, setLoraR] = useState(16);
  const [loraAlpha, setLoraAlpha] = useState(32);
  const [gradAccum, setGradAccum] = useState(4);
  const [multiGpu, setMultiGpu] = useState(false);
  const [activeTab, setActiveTab] = useState<TrainStudioTab>("setup");
  const [distributedStrategy, setDistributedStrategy] = useState("auto");
  const [distributedNproc, setDistributedNproc] = useState("");
  const [distributedNodes, setDistributedNodes] = useState(1);
  const [distributedNodeRank, setDistributedNodeRank] = useState(0);
  const [distributedMasterAddr, setDistributedMasterAddr] = useState("127.0.0.1");
  const [distributedMasterPort, setDistributedMasterPort] = useState(29500);
  const [ddpBackend, setDdpBackend] = useState("");
  const [ddpFindUnused, setDdpFindUnused] = useState(false);
  const [distributedOverridesEnabled, setDistributedOverridesEnabled] = useState(false);
  const [distributedBatchSize, setDistributedBatchSize] = useState(2);
  const [distributedGradAccum, setDistributedGradAccum] = useState(4);
  const [distributedLearningRate, setDistributedLearningRate] = useState(0.0002);
  const [distributedMaxSeq, setDistributedMaxSeq] = useState(2048);
  const [distributedEpochs, setDistributedEpochs] = useState(5);
  const [distributedLoggingSteps, setDistributedLoggingSteps] = useState(10);
  const [distributedSaveSteps, setDistributedSaveSteps] = useState(100);
  const [distributedMaxEvalSamples, setDistributedMaxEvalSamples] = useState(128);
  const [cloudGpuEnabled, setCloudGpuEnabled] = useState(false);
  const [cloudGpuProvider, setCloudGpuProvider] = useState("none");
  const [cloudGpuRegion, setCloudGpuRegion] = useState("");
  const [cloudGpuInstanceType, setCloudGpuInstanceType] = useState("");
  const [cloudGpuCount, setCloudGpuCount] = useState(1);
  const [cloudGpuProject, setCloudGpuProject] = useState("");
  const [cloudCredentials, setCloudCredentials] = useState<CloudGpuCredential[]>([]);
  const [selectedCloudCredentialId, setSelectedCloudCredentialId] = useState("");
  const [cloudCredentialName, setCloudCredentialName] = useState("");
  const [cloudAuthKind, setCloudAuthKind] = useState("api_key");
  const [cloudApiKey, setCloudApiKey] = useState("");
  const [cloudAccessKeyId, setCloudAccessKeyId] = useState("");
  const [cloudSecretAccessKey, setCloudSecretAccessKey] = useState("");
  const [cloudSessionToken, setCloudSessionToken] = useState("");
  const [cloudSshUsername, setCloudSshUsername] = useState("");
  const [cloudSshPrivateKey, setCloudSshPrivateKey] = useState("");
  const [cloudBootstrapCommand, setCloudBootstrapCommand] = useState("");
  const [cloudCredentialMsg, setCloudCredentialMsg] = useState("");
  const [useFusedKernels, setUseFusedKernels] = useState(true);
  const [useFusedCe, setUseFusedCe] = useState(true);
  const [gradCkpt, setGradCkpt] = useState(true);
  const [trainResponsesOnly, setTrainResponsesOnly] = useState(true);
  const [useRsLora, setUseRsLora] = useState(false);
  const [packing, setPacking] = useState(false);
  const [exportOnComplete, setExportOnComplete] = useState(true);
  const [exportProfile, setExportProfile] = useState("lora_bundle");
  const [exportQuants, setExportQuants] = useState<string[]>(["q4_k_m", "q8_0", "f16"]);
  const [exportProfiles, setExportProfiles] = useState<
    { id: string; formats: string[]; default_gguf_quants: string[] }[]
  >([]);
  const [logs, setLogs] = useState<string[]>([]);
  const [activeJob, setActiveJob] = useState<string | null>(null);
  const [jobStatus, setJobStatus] = useState<string | null>(null);
  const [trainingMetrics, setTrainingMetrics] = useState<TrainingMetricPoint[]>([]);
  const [systemMetrics, setSystemMetrics] = useState<SystemMetrics[]>([]);
  const [metricsOpen, setMetricsOpen] = useState(false);
  const [starting, setStarting] = useState(false);
  const [downloadingModel, setDownloadingModel] = useState(false);
  const [loadProgress, setLoadProgress] = useState<ModelProgressState | null>(null);
  const [downloadError, setDownloadError] = useState<string | null>(null);
  const [hwApplied, setHwApplied] = useState(false);
  const [recommendations, setRecommendations] = useState<TrainingRecommendations | null>(null);
  const [recLoading, setRecLoading] = useState(false);
  const [configCustomized, setConfigCustomized] = useState(false);
  const [datasetValid, setDatasetValid] = useState(true);
  const [datasetError, setDatasetError] = useState<string | null>(null);
  const [analyzingDataset, setAnalyzingDataset] = useState(false);
  const [datasetAnalysis, setDatasetAnalysis] = useState<DatasetAnalysis | null>(null);
  const lastDatasetAnalysisRef = useRef<{
    dataset: string;
    requestedFormat: string;
    resolvedFormat: string;
  } | null>(null);

  const analyzeDataset = useCallback(async (ds: string, fmt: string) => {
    if (!ds) {
      setDatasetValid(true);
      setDatasetError(null);
      setDatasetAnalysis(null);
      return;
    }
    setAnalyzingDataset(true);
    try {
      const res = await api.analyzeDataset(ds, fmt);
      if (res.valid) {
        setDatasetValid(true);
        setDatasetError(null);
        setDatasetAnalysis(res);
        lastDatasetAnalysisRef.current = {
          dataset: ds,
          requestedFormat: fmt,
          resolvedFormat: res.resolved_format,
        };
        const rec = res.recommended_config;
        if (rec?.dataset_format && rec.dataset_format !== "auto") {
          setDatasetFormat(rec.dataset_format);
        }
        if (rec?.train_on_responses_only != null) {
          setTrainResponsesOnly(rec.train_on_responses_only);
        }
        if (rec?.max_seq_length) setMaxSeq(rec.max_seq_length);
        if (rec?.epochs) setEpochs(rec.epochs);
        if (rec?.early_stopping != null) setEarlyStopping(rec.early_stopping);
        if (rec?.early_stopping_patience != null) {
          setEarlyStoppingPatience(rec.early_stopping_patience);
        }
        if (rec?.preprocess_dataset != null) setPreprocessDataset(rec.preprocess_dataset);
        if (rec?.packing != null) setPacking(rec.packing);
      } else {
        setDatasetValid(false);
        setDatasetError(res.error || "Dataset cannot be normalized for training");
        setDatasetAnalysis(null);
      }
    } catch (e: any) {
      setDatasetValid(false);
      setDatasetError(e?.message || "Failed to analyze dataset");
      setDatasetAnalysis(null);
    } finally {
      setAnalyzingDataset(false);
    }
  }, []);
  const downloadGenRef = useRef(0);
  const sseAbortRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    return () => {
      sseAbortRef.current?.();
    };
  }, []);

  useEffect(() => {
    api.listTrainingJobs().then(setJobs).catch(console.error);
    api.listExportProfiles().then(setExportProfiles).catch(console.error);
    api.listCloudGpuCredentials().then(setCloudCredentials).catch(console.error);
    if (pendingModel) setModelId(pendingModel);
  }, [pendingModel]);

  useEffect(() => {
    // Restore a prior user choice only — never auto-pick the first local model on tab open
    // (that would kick off recommendations / analysis before any selection).
    if (pendingModel || modelId) return;
    const stored = readStoredModel("train:model");
    if (stored) setModelId(stored);
  }, [pendingModel, modelId]);

  useEffect(() => {
    if (!pendingModel || !pendingDownload) return;
    const downloadGen = ++downloadGenRef.current;
    let cancelled = false;

    const run = async () => {
      setDownloadingModel(true);
      setDownloadError(null);
      setLoadProgress(initialDownloadProgress(pendingModel, pendingDownloadBytes));
      try {
        if (isTrainModelCached(localModels, pendingModel)) {
          setModelId(pendingModel);
          setSearchParams({ model: pendingModel }, { replace: true });
          return;
        }
        await ensureTrainHubModel(pendingModel, setLoadProgress, pendingDownloadBytes);
        if (cancelled) return;
        invalidateApiCache("/training/models");
        invalidateApiCache("/inference/models");
        await refreshLocalModels();
        setModelId(pendingModel);
        setSearchParams({ model: pendingModel }, { replace: true });
      } catch (e) {
        if (cancelled) return;
        setDownloadError(e instanceof Error ? e.message : "Failed to download model from Hugging Face");
      } finally {
        if (downloadGen === downloadGenRef.current) {
          setDownloadingModel(false);
          setLoadProgress(null);
        }
      }
    };

    void run();
    return () => {
      cancelled = true;
    };
  }, [pendingModel, pendingDownload, pendingDownloadBytes, localModels, refreshLocalModels, setSearchParams]);

  useEffect(() => {
    if (!hw || hwApplied || pendingModel) return;
    const d = hw.training_defaults;
    if (!d) return;
    setBatchSize(d.batch_size);
    setGradAccum(d.gradient_accumulation_steps);
    setMaxSeq(d.max_seq_length);
    setQuant(d.quant);
    setMethod(d.method);
    setGradCkpt(d.gradient_checkpointing);
    if (d.use_fused_kernels != null) setUseFusedKernels(d.use_fused_kernels);
    if (d.use_fused_ce != null) setUseFusedCe(d.use_fused_ce);
    setHwApplied(true);
  }, [hw, hwApplied, pendingModel]);

  useEffect(() => {
    if (method === "full") setExportProfile("full_bundle");
    else if (method === "lora") setExportProfile("lora_bundle");
  }, [method]);

  useEffect(() => {
    // Wait for an explicit model or dataset choice — do not run recs/analysis on bare tab open.
    if (!modelId.trim() && !dataset.trim()) {
      setRecommendations(null);
      setRecLoading(false);
      return;
    }
    let cancelled = false;
    setRecLoading(true);
    api
      .getTrainingRecommendations(modelId, dataset)
      .then((rec) => {
        if (!cancelled) setRecommendations(rec);
      })
      .catch(() => {
        if (!cancelled) setRecommendations(null);
      })
      .finally(() => {
        if (!cancelled) setRecLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [modelId, dataset]);

  // Analyze the full dataset schema only after the user selects a dataset (not on tab open).
  useEffect(() => {
    if (!dataset.trim()) {
      lastDatasetAnalysisRef.current = null;
      setDatasetValid(true);
      setDatasetError(null);
      setDatasetAnalysis(null);
      setAnalyzingDataset(false);
      return;
    }
    const last = lastDatasetAnalysisRef.current;
    if (
      last?.dataset === dataset &&
      (last.requestedFormat === datasetFormat || last.resolvedFormat === datasetFormat)
    ) {
      return;
    }
    const t = setTimeout(() => {
      analyzeDataset(dataset, datasetFormat);
    }, 350);
    return () => clearTimeout(t);
  }, [dataset, datasetFormat, analyzeDataset]);

  useEffect(() => {
    if (configCustomized || !recommendations?.config || datasetAnalysis) return;
    const rec = recommendations.config;
    if (rec.dataset_format && rec.dataset_format !== "auto") {
      setDatasetFormat(rec.dataset_format);
    }
    if (rec.train_on_responses_only != null) {
      setTrainResponsesOnly(rec.train_on_responses_only);
    }
  }, [recommendations, configCustomized, datasetAnalysis]);

  const applyRecommendations = useCallback(() => {
    const rec = recommendations?.config;
    if (!rec) return;
    setMethod(rec.method);
    setQuant(rec.quant);
    setBatchSize(rec.batch_size);
    setGradAccum(rec.gradient_accumulation_steps);
    setMaxSeq(rec.max_seq_length);
    setLr(rec.learning_rate);
    setEpochs(rec.epochs);
    if (rec.early_stopping != null) setEarlyStopping(rec.early_stopping);
    if (rec.early_stopping_patience != null) setEarlyStoppingPatience(rec.early_stopping_patience);
    if (rec.preprocess_dataset != null) setPreprocessDataset(rec.preprocess_dataset);
    setLoraR(rec.lora_r);
    setLoraAlpha(rec.lora_alpha);
    setGradCkpt(rec.gradient_checkpointing);
    setUseFusedKernels(rec.use_triton);
    setUseFusedCe(rec.use_fused_ce);
    setTrainResponsesOnly(rec.train_on_responses_only);
    setUseRsLora(rec.use_rslora);
    setPacking(rec.packing);
    if (rec.dataset_format && rec.dataset_format !== "auto") {
      setDatasetFormat(rec.dataset_format);
    }
    setConfigCustomized(false);
  }, [recommendations]);

  const modelBlocked = Boolean(modelId && recommendations?.trainable === false);
  const canStart =
    Boolean(modelId.trim() && dataset.trim()) &&
    !modelBlocked &&
    datasetValid &&
    !analyzingDataset;

  useEffect(() => {
    if (!exportProfiles.length) return;
    const prof = exportProfiles.find((p) => p.id === exportProfile);
    if (prof?.default_gguf_quants?.length) {
      setExportQuants(prof.default_gguf_quants);
    }
  }, [exportProfile, exportProfiles]);

  useEffect(() => {
    if (distributedOverridesEnabled) return;
    setDistributedBatchSize(batchSize);
    setDistributedGradAccum(gradAccum);
    setDistributedLearningRate(lr);
    setDistributedMaxSeq(maxSeq);
    setDistributedEpochs(epochs);
  }, [batchSize, gradAccum, lr, maxSeq, epochs, distributedOverridesEnabled]);

  const toggleExportQuant = (quant: string) => {
    setExportQuants((prev) =>
      prev.includes(quant) ? prev.filter((q) => q !== quant) : [...prev, quant],
    );
  };

  const GGUF_QUANT_OPTIONS = ["q2_k", "q3_k_m", "q4_k_m", "q5_k_m", "q6_k", "q8_0", "f16"];
  const LOCAL_GPU_COUNT = hw?.gpus?.length ?? 0;
  const distributedEnabled = multiGpu || distributedStrategy === "ddp";
  const distributedWorldSize =
    Math.max(1, Number(distributedNproc || LOCAL_GPU_COUNT || 1)) *
    Math.max(1, distributedNodes);
  const distributedEffectiveBatch =
    distributedWorldSize *
    Math.max(1, distributedOverridesEnabled ? distributedBatchSize : batchSize) *
    Math.max(1, distributedOverridesEnabled ? distributedGradAccum : gradAccum);
  const safeCloudLabel = (value: string) =>
    !/(token|secret|password|apikey|api_key|:\/\/)/i.test(value);
  const cloudConfigValid =
    !cloudGpuEnabled ||
    (Boolean(selectedCloudCredentialId) &&
      cloudGpuProvider !== "none" &&
      cloudGpuInstanceType.trim().length > 0 &&
      [cloudGpuRegion, cloudGpuInstanceType, cloudGpuProject].every(safeCloudLabel));

  const jobStatusBadge = (status: string) => (
    <span className={`badge badge-${status}`}>{status}</span>
  );

  const trainingModelLabel = (job: TrainingJob) => {
    try {
      const cfg = JSON.parse(job.config_json) as { model_id?: string };
      const id = cfg.model_id;
      return id ? id.split("/").pop() || id : "—";
    } catch {
      return "—";
    }
  };

  const ensureModelCached = async () => {
    if (isTrainModelCached(localModels, modelId)) return;
    setDownloadingModel(true);
    setDownloadError(null);
    setLoadProgress(initialDownloadProgress(modelId));
    try {
      await ensureTrainHubModel(modelId, setLoadProgress);
      invalidateApiCache("/training/models");
      invalidateApiCache("/inference/models");
      await refreshLocalModels();
    } finally {
      setDownloadingModel(false);
      setLoadProgress(null);
    }
  };

  const refreshCloudCredentials = async () => {
    const rows = await api.listCloudGpuCredentials();
    setCloudCredentials(rows);
    return rows;
  };

  const saveCloudCredential = async () => {
    if (!cloudCredentialName.trim() || cloudGpuProvider === "none") return;
    try {
      const saved = await api.saveCloudGpuCredential({
        name: cloudCredentialName.trim(),
        provider: cloudGpuProvider,
        auth_kind: cloudAuthKind,
        api_key: cloudApiKey,
        access_key_id: cloudAccessKeyId,
        secret_access_key: cloudSecretAccessKey,
        session_token: cloudSessionToken,
        ssh_username: cloudSshUsername,
        ssh_private_key: cloudSshPrivateKey,
        bootstrap_command: cloudBootstrapCommand,
        region: cloudGpuRegion,
        project: cloudGpuProject,
      });
      setCloudCredentialMsg("Cloud access saved encrypted locally.");
      setSelectedCloudCredentialId(saved.id);
      setCloudCredentialName("");
      setCloudApiKey("");
      setCloudAccessKeyId("");
      setCloudSecretAccessKey("");
      setCloudSessionToken("");
      setCloudSshPrivateKey("");
      setCloudBootstrapCommand("");
      await refreshCloudCredentials();
    } catch (err) {
      setCloudCredentialMsg((err as Error).message);
    }
  };

  const deleteCloudCredential = async (credentialId: string) => {
    try {
      await api.deleteCloudGpuCredential(credentialId);
      if (selectedCloudCredentialId === credentialId) setSelectedCloudCredentialId("");
      setCloudCredentialMsg("Cloud access credential deleted.");
      await refreshCloudCredentials();
    } catch (err) {
      setCloudCredentialMsg((err as Error).message);
    }
  };

  const start = async () => {
    if (modelBlocked) {
      setDownloadError(recommendations?.warnings[0] || "This model cannot be trained on your hardware.");
      return;
    }
    if (!datasetValid) {
      setDownloadError(datasetError || "Dataset cannot be normalized for training");
      return;
    }
    if (!cloudConfigValid) {
      setDownloadError("Select a saved cloud access credential and keep cloud target labels free of URLs or secret-looking text.");
      setActiveTab("cloud");
      return;
    }
    setStarting(true);
    setLogs([]);
    setTrainingMetrics([]);
    setSystemMetrics([]);
    setJobStatus("running");
    setDownloadError(null);
    try {
      await ensureModelCached();

      const exportPayload =
        exportOnComplete && method !== "embedding"
          ? {
              profile: exportProfile,
              gguf_quantizations: exportQuants.length ? exportQuants : undefined,
            }
          : undefined;

      const baseTrainingConfig: Record<string, unknown> = {
        model_id: modelId,
        dataset,
        method,
        quant,
        dataset_format: datasetFormat,
        epochs,
        batch_size: batchSize,
        learning_rate: lr,
        max_seq_length: maxSeq,
        preprocess_dataset: preprocessDataset,
        deduplicate_dataset: preprocessDataset,
        early_stopping: earlyStopping,
        early_stopping_patience: earlyStoppingPatience,
        lora_r: loraR,
        lora_alpha: loraAlpha,
        gradient_accumulation_steps: gradAccum,
        gradient_checkpointing: gradCkpt,
        use_triton: useFusedKernels,
        use_fused_ce: useFusedCe,
        train_on_responses_only: trainResponsesOnly,
        use_rslora: useRsLora,
        packing,
        output_dir: "./outputs",
      };

      const distributedTrainingOverrides =
        distributedEnabled && distributedOverridesEnabled
          ? {
              epochs: distributedEpochs,
              batch_size: distributedBatchSize,
              learning_rate: distributedLearningRate,
              max_seq_length: distributedMaxSeq,
              gradient_accumulation_steps: distributedGradAccum,
              logging_steps: distributedLoggingSteps,
              save_steps: distributedSaveSteps,
              max_eval_samples: distributedMaxEvalSamples,
            }
          : {};

      const res = await api.startTraining(
        {
          ...baseTrainingConfig,
          ...distributedTrainingOverrides,
          multi_gpu: distributedEnabled,
          distributed_strategy: distributedStrategy,
          distributed_nproc_per_node: distributedNproc ? Number(distributedNproc) : undefined,
          distributed_num_nodes: distributedNodes,
          distributed_node_rank: distributedNodeRank,
          distributed_master_addr: distributedMasterAddr,
          distributed_master_port: distributedMasterPort,
          ddp_backend: ddpBackend || undefined,
          ddp_find_unused_parameters: ddpFindUnused,
          cloud_gpu_enabled: cloudGpuEnabled,
          cloud_gpu_provider: cloudGpuEnabled ? cloudGpuProvider : "none",
          cloud_gpu_region: cloudGpuEnabled ? cloudGpuRegion.trim() : "",
          cloud_gpu_instance_type: cloudGpuEnabled ? cloudGpuInstanceType.trim() : "",
          cloud_gpu_count: cloudGpuEnabled ? cloudGpuCount : undefined,
          cloud_gpu_project: cloudGpuEnabled ? cloudGpuProject.trim() : "",
          cloud_gpu_credential_id: cloudGpuEnabled ? selectedCloudCredentialId : undefined,
        },
        distributedEnabled,
        exportPayload,
        datasetAnalysis?.analysis_token,
      );
      setActiveJob(res.job_id);
      setMetricsOpen(true);
      sseAbortRef.current?.();
      sseAbortRef.current = subscribeSSE(
        `/training/jobs/${res.job_id}/stream`,
        (event, data) => {
          if (event === "log") setLogs((l) => appendBoundedLog(l, data));
          if (event === "error") setLogs((l) => appendBoundedLog(l, `ERROR: ${data}`));
          if (event === "metric") {
            try {
              const point = JSON.parse(data) as TrainingMetricPoint & SystemMetrics;
              if (point.type === "system") {
                setSystemMetrics((prev) => [...prev.slice(-499), point as SystemMetrics]);
              } else {
                setTrainingMetrics((prev) => [...prev.slice(-1999), point]);
              }
            } catch {
              /* ignore malformed metric payloads */
            }
          }
          if (event === "status") {
            setJobStatus(data);
            if (data === "completed" || data === "failed" || data === "cancelled") {
              invalidateApiCache("/training/models");
              invalidateApiCache("/inference/models");
              refreshLocalModels().catch(console.error);
            }
            api.listTrainingJobs().then(setJobs);
          }
        },
        (err) => setLogs((l) => appendBoundedLog(l, `ERROR: ${err.message}`)),
      );
      api.listTrainingJobs().then(setJobs);
    } catch (e) {
      setDownloadError(e instanceof Error ? e.message : "Failed to prepare model for training");
      setJobStatus(null);
    } finally {
      setStarting(false);
    }
  };

  return (
    <StudioPageShell
      title="Training Studio"
      subtitle="Fine-tune with LoRA/QLoRA. The full dataset is analyzed to set format and hyperparameters (independent of chat)."
      banner={
        hw?.training_defaults ? (
          <div className="hw-inline-banner card">
            <span className="trust-badge">{hw.tier_label}</span>
            <span className="muted-text">{hw.training_defaults.note}</span>
          </div>
        ) : undefined
      }
    >
      <TrainingMetricsDashboard
        jobId={activeJob}
        open={metricsOpen}
        onClose={() => setMetricsOpen(false)}
        trainingPoints={trainingMetrics}
        systemPoints={systemMetrics}
        status={jobStatus}
      />

      {(loadProgress || downloadingModel) && (
        <div className="card studio-card studio-download-card">
          <StudioCardHeader
            icon="↓"
            title="Downloading base model"
            description="Fetching safetensors snapshot from Hugging Face Hub"
          />
          <ModelLoadProgress progress={loadProgress} modelName={modelId.split("/").pop()} />
        </div>
      )}
      {downloadError && (
        <div className="status-callout status-callout-error studio-error-callout" role="alert">
          <div className="status-callout-body">
            <strong className="status-callout-title">
              {downloadingModel || /download/i.test(downloadError) ? "Download failed" : "Training error"}
            </strong>
            <div className="status-callout-text">{downloadError}</div>
          </div>
        </div>
      )}

      <Tabs<TrainStudioTab>
        className="train-tab-bar"
        value={activeTab}
        onChange={setActiveTab}
        aria-label="Training sections"
        items={[
          {
            id: "setup",
            label: "Setup",
            description: "Model, data, hyperparameters, export",
            icon: "①",
          },
          {
            id: "distributed",
            label: "Distributed",
            description: "Accelerate DDP and cloud GPU launch settings",
            icon: "②",
            badge: distributedEnabled ? "on" : undefined,
          },
          {
            id: "cloud",
            label: "Cloud access",
            description: "Encrypted provider keys, SSH, bootstrap",
            icon: "③",
            badge: selectedCloudCredentialId ? "saved" : undefined,
          },
        ]}
      />

      {activeTab === "setup" && (
      <div className="train-layout train-layout--studio train-layout--studio-fit tab-panel">
        <div className="card studio-card">
          <StudioCardHeader
            icon="①"
            title="Model & data"
            description="Base checkpoint and training dataset"
          />
          <StudioCardBody>
          <div className="form-field">
            <label>Base model</label>
            <HfBaseModelPicker
              mode="train"
              value={modelId}
              localModels={localModels}
              disabled={downloadingModel}
              onChange={(value) => {
                setModelId(value);
                writeStoredModel("train:model", value);
                setConfigCustomized(false);
              }}
            />
          </div>
          {modelBlocked && (
            <div className="status-callout status-callout-warn studio-error-callout" role="alert">
              <div className="status-callout-body">
                <strong className="status-callout-title">Not trainable</strong>
                <div className="status-callout-text">
                  {recommendations?.warnings[0] || "This model cannot be trained on your hardware."}
                </div>
              </div>
            </div>
          )}
          {(recommendations || recLoading) && !modelBlocked && (
            <div className="status-callout status-callout-info train-rec-panel">
              <div className="status-callout-body">
                <strong className="status-callout-title">
                  {recLoading ? "Analyzing model & dataset…" : "Suggested settings"}
                </strong>
                {!recLoading && recommendations && (
                  <>
                    <div className="status-callout-text">
                      {recommendations.notes.slice(0, 2).join(" ")}
                      {recommendations.est_training_vram_gb != null && (
                        <> Est. training VRAM ~{recommendations.est_training_vram_gb} GB.</>
                      )}
                    </div>
                    <div className="train-rec-actions">
                      <button
                        type="button"
                        className="btn btn-sm"
                        onClick={applyRecommendations}
                        disabled={recLoading}
                      >
                        Apply recommended settings
                      </button>
                      {configCustomized && (
                        <span className="muted-text studio-field-hint-compact">You changed settings manually.</span>
                      )}
                    </div>
                  </>
                )}
              </div>
            </div>
          )}
          {localModels.length > 0 && (
            <p className="muted-text studio-field-hint studio-field-hint-compact">
              {localModels.length} snapshot{localModels.length === 1 ? "" : "s"} cached locally.
            </p>
          )}
          <div className="form-field">
            <label>Dataset</label>
            <HfDatasetPicker value={dataset} onChange={(v) => { setDataset(v); setConfigCustomized(false); }} />
            {!dataset.trim() && (
              <p className="muted-text studio-field-hint studio-field-hint-compact">
                Select a dataset to analyze schema and prepare training settings. Nothing runs until you pick one.
              </p>
            )}
          </div>
          <div className="form-field">
            <label>Dataset format</label>
            <select
              value={datasetFormat}
              onChange={(e) => {
                setDatasetFormat(e.target.value);
                setConfigCustomized(true);
              }}
            >
              <option value="auto">Auto-detect</option>
              <option value="chat">Chat / messages</option>
              <option value="preference">Preference (chosen/rejected)</option>
              <option value="alpaca">Alpaca (instruction/output)</option>
              <option value="sharegpt">ShareGPT conversations</option>
              <option value="text">Plain text</option>
            </select>
            {recommendations?.config.dataset_format &&
              recommendations.config.dataset_format !== "auto" &&
              datasetFormat === "auto" && (
                <p className="muted-text studio-field-hint studio-field-hint-compact">
                  Recommended: {recommendations.config.dataset_format} — apply suggested settings to use it.
                </p>
              )}
          </div>

          {(analyzingDataset || !datasetValid || datasetError) && (
            <div className={`status-callout ${datasetValid ? "status-callout-warn" : "status-callout-error"} studio-error-callout`} style={{marginTop: 8}}>
              <div className="status-callout-body">
                <strong className="status-callout-title">
                  {analyzingDataset ? "Analyzing dataset…" : datasetValid ? "Dataset warning" : "Dataset error"}
                </strong>
                <div className="status-callout-text">
                  {analyzingDataset
                    ? "Scanning the entire dataset, detecting schema, and preparing research-grade training settings..."
                    : (datasetError || "Dataset looks invalid for training.")}
                </div>
                {!datasetValid && (
                  <div className="muted-text" style={{fontSize: "12px", marginTop: 4}}>
                    Pick a dataset with instruction/output pairs, Q&A columns, multi-turn messages, preference pairs, or plain text/code fields.
                  </div>
                )}
              </div>
            </div>
          )}

          {datasetAnalysis?.valid && !analyzingDataset && (
            <div className="status-callout status-callout-info train-rec-panel" style={{ marginTop: 8 }}>
              <div className="status-callout-body">
                <strong className="status-callout-title">Dataset analysis</strong>
                <div className="status-callout-text">
                  {datasetAnalysis.domain_label} · {datasetAnalysis.resolved_format} format ·{" "}
                  {datasetAnalysis.kept.toLocaleString()} / {datasetAnalysis.initial_samples.toLocaleString()} rows retained (
                  {datasetAnalysis.utilization_pct}%)
                </div>
                <div className="muted-text studio-field-hint studio-field-hint-compact">
                  Columns: {datasetAnalysis.columns.join(", ")} · p95 ~{datasetAnalysis.length_stats.estimated_tokens_p95} tokens
                </div>
                {datasetAnalysis.notes[0] && (
                  <div className="muted-text studio-field-hint studio-field-hint-compact">{datasetAnalysis.notes[0]}</div>
                )}
                {datasetAnalysis.sample_preview.length > 0 && (
                  <details className="train-dataset-preview" style={{ marginTop: 8 }}>
                    <summary className="muted-text">Preview normalized rows</summary>
                    <pre className="train-dataset-preview-body">
                      {JSON.stringify(datasetAnalysis.sample_preview, null, 2)}
                    </pre>
                  </details>
                )}
              </div>
            </div>
          )}
          </StudioCardBody>
        </div>

        <div className="card studio-card">
          <StudioCardHeader
            icon="②"
            title="Hyperparameters"
            description="Method, quantization, and core training knobs"
          />
          <StudioCardBody>

          <div className="option-grid">
            <div className="form-field">
              <label>Method</label>
              <select value={method} onChange={(e) => { setMethod(e.target.value); setConfigCustomized(true); }}>
                <option value="lora">LoRA / QLoRA</option>
                <option value="full">Full fine-tune</option>
                <option value="embedding">Embedding</option>
              </select>
            </div>
            <div className="form-field">
              <label>Quantization</label>
              <select value={quant} onChange={(e) => { setQuant(e.target.value); setConfigCustomized(true); }}>
                <option value="4bit">4-bit (QLoRA)</option>
                <option value="8bit">8-bit</option>
                <option value="16bit">16-bit FP16</option>
                <option value="none">None (full precision)</option>
              </select>
            </div>
          </div>
          <div className="studio-slider-grid">
            <div className="slider-row">
              <label>Max epochs: {epochs}</label>
              <input type="range" min={1} max={20} value={epochs} onChange={(e) => { setEpochs(+e.target.value); setConfigCustomized(true); }} />
              <p className="muted-text studio-field-hint studio-field-hint-compact">
                Training stops earlier when early stopping finds the best eval loss.
              </p>
            </div>
            <div className="slider-row">
              <label>Batch size: {batchSize}</label>
              <input type="range" min={1} max={16} value={batchSize} onChange={(e) => { setBatchSize(+e.target.value); setConfigCustomized(true); }} />
            </div>
            <div className="slider-row">
              <label>Max seq length: {maxSeq}</label>
              <input type="range" min={512} max={8192} step={256} value={maxSeq} onChange={(e) => { setMaxSeq(+e.target.value); setConfigCustomized(true); }} />
            </div>
            <div className="slider-row">
              <label>Learning rate: {lr.toExponential(1)}</label>
              <input type="range" min={-6} max={-3} step={0.1} value={Math.log10(lr)} onChange={(e) => { setLr(10 ** +e.target.value); setConfigCustomized(true); }} />
            </div>
          </div>

          {method === "lora" && (
            <FormSection title="LoRA settings" hint="Rank, alpha, and gradient accumulation." collapsible defaultOpen={false}>
              <div className="studio-slider-grid">
                <div className="slider-row">
                  <label>LoRA rank (r): {loraR}</label>
                  <input type="range" min={4} max={128} step={4} value={loraR} onChange={(e) => setLoraR(+e.target.value)} />
                </div>
                <div className="slider-row">
                  <label>LoRA alpha: {loraAlpha}</label>
                  <input type="range" min={8} max={256} step={8} value={loraAlpha} onChange={(e) => setLoraAlpha(+e.target.value)} />
                </div>
              </div>
              <div className="slider-row">
                <label>Grad accumulation: {gradAccum}</label>
                <input type="range" min={1} max={32} value={gradAccum} onChange={(e) => { setGradAccum(+e.target.value); setConfigCustomized(true); }} />
              </div>
            </FormSection>
          )}
          </StudioCardBody>
        </div>

        <div className="card studio-card">
          <StudioCardHeader
            icon="③"
            title="Optimization & export"
            description="Memory trade-offs, post-training bundles, and run controls"
            meta={
              activeJob && jobStatus ? (
                <span className={`badge badge-${jobStatus}`}>{jobStatus}</span>
              ) : undefined
            }
          />
          <StudioCardBody>

          <FormSection title="Optimization" hint="Memory and throughput trade-offs." collapsible defaultOpen={false}>
            <div className="studio-checkbox-grid">
              <label className="studio-checkbox-item">
                <input type="checkbox" checked={preprocessDataset} onChange={(e) => { setPreprocessDataset(e.target.checked); setConfigCustomized(true); }} />
                Normalize &amp; clean dataset
              </label>
              <label className="studio-checkbox-item">
                <input type="checkbox" checked={earlyStopping} onChange={(e) => { setEarlyStopping(e.target.checked); setConfigCustomized(true); }} />
                Early stopping (optimal epoch)
              </label>
              <label className="studio-checkbox-item">
                <input type="checkbox" checked={gradCkpt} onChange={(e) => { setGradCkpt(e.target.checked); setConfigCustomized(true); }} />
                Gradient checkpointing
              </label>
              <label className="studio-checkbox-item">
                <input type="checkbox" checked={trainResponsesOnly} onChange={(e) => { setTrainResponsesOnly(e.target.checked); setConfigCustomized(true); }} />
                Train on responses only
              </label>
              <label className="studio-checkbox-item">
                <input type="checkbox" checked={useRsLora} onChange={(e) => { setUseRsLora(e.target.checked); setConfigCustomized(true); }} />
                Rank-stabilized LoRA (rsLoRA)
              </label>
              <label className="studio-checkbox-item">
                <input type="checkbox" checked={packing} onChange={(e) => { setPacking(e.target.checked); setConfigCustomized(true); }} />
                Sequence packing
              </label>
              <label className="studio-checkbox-item">
                <input type="checkbox" checked={distributedEnabled} readOnly />
                Distributed configured
              </label>
              <label
                className="studio-checkbox-item"
                title={hw?.training_defaults?.kernel_backend ? `Backend: ${hw.training_defaults.kernel_backend}` : undefined}
              >
                <input
                  type="checkbox"
                  checked={useFusedKernels}
                  onChange={(e) => setUseFusedKernels(e.target.checked)}
                  disabled={hw?.training_defaults?.use_fused_kernels === false}
                />
                Fused kernels (RMSNorm + SwiGLU)
              </label>
              <label className="studio-checkbox-item">
                <input
                  type="checkbox"
                  checked={useFusedCe}
                  onChange={(e) => setUseFusedCe(e.target.checked)}
                  disabled={hw?.training_defaults?.use_fused_ce === false}
                />
                Fused cross-entropy
              </label>
            </div>
            {earlyStopping && (
              <div className="slider-row">
                <label>Early-stop patience: {earlyStoppingPatience}</label>
                <input
                  type="range"
                  min={1}
                  max={10}
                  value={earlyStoppingPatience}
                  onChange={(e) => setEarlyStoppingPatience(+e.target.value)}
                />
              </div>
            )}
          </FormSection>

          {method !== "embedding" && (
            <FormSection
              title="Post-training export"
              hint="Bundle LoRA, merged weights, and GGUF quants when the run finishes."
              collapsible
              defaultOpen={false}
            >
              <label className="studio-checkbox-item studio-checkbox-item-standalone">
                <input
                  type="checkbox"
                  checked={exportOnComplete}
                  onChange={(e) => setExportOnComplete(e.target.checked)}
                />
                Export when training completes
              </label>
              {exportOnComplete && (
                <>
                  <div className="form-field">
                    <label>Export profile</label>
                    <select value={exportProfile} onChange={(e) => setExportProfile(e.target.value)}>
                      {exportProfiles.map((p) => (
                        <option key={p.id} value={p.id}>
                          {p.id} ({p.formats.join(", ")})
                        </option>
                      ))}
                      {!exportProfiles.length && (
                        <>
                          <option value="lora_bundle">lora_bundle</option>
                          <option value="lora_adapter">lora_adapter</option>
                          <option value="full_bundle">full_bundle</option>
                        </>
                      )}
                    </select>
                  </div>
                  {(exportProfile.includes("gguf") || exportProfile.includes("bundle")) && (
                    <div className="form-field">
                      <label>GGUF quantizations</label>
                      <div className="studio-checkbox-grid studio-checkbox-grid-compact">
                        {GGUF_QUANT_OPTIONS.map((q) => (
                          <label key={q} className="studio-checkbox-item">
                            <input
                              type="checkbox"
                              checked={exportQuants.includes(q)}
                              onChange={() => toggleExportQuant(q)}
                            />
                            {q}
                          </label>
                        ))}
                      </div>
                    </div>
                  )}
                </>
              )}
            </FormSection>
          )}

          </StudioCardBody>
          <div className="studio-action-bar studio-action-bar-flush">
            <button
              className="btn btn-primary btn-lg"
              onClick={start}
              disabled={starting || downloadingModel || !canStart}
            >
              {starting
                ? "Starting…"
                : downloadingModel
                  ? "Downloading model…"
                  : !modelId.trim() || !dataset.trim()
                    ? "Select model and dataset"
                    : "Start training"}
            </button>
            {activeJob && (
              <button type="button" className="btn" onClick={() => setMetricsOpen(true)}>
                View metrics
              </button>
            )}
          </div>
        </div>

        <div className="card studio-card studio-card-scroll">
          <StudioCardHeader
            icon="④"
            title="Live output"
            description="Streaming logs from the active training job"
            tone="monitor"
            meta={
              activeJob ? <span className="mono studio-job-id">{activeJob.slice(0, 8)}…</span> : undefined
            }
          />
          <LogStream
            logs={logs}
            emptyMessage="Start a training run to stream logs here."
            fill
            label="Training log"
          />
        </div>

        <div className="card studio-card studio-card-compact">
          <StudioCardHeader
            icon="⑤"
            title="History"
            description="Past jobs on this machine"
            tone="history"
            meta={
              jobs.length > 0 ? (
                <span className="badge badge-dim">{jobs.length}</span>
              ) : undefined
            }
          />
          <DataTable
            columns={[
              {
                key: "id",
                header: "Job",
                mono: true,
                render: (j) => `${j.id.slice(0, 8)}…`,
              },
              {
                key: "status",
                header: "Status",
                render: (j) => jobStatusBadge(j.status),
              },
              {
                key: "model",
                header: "Model",
                render: (j) => trainingModelLabel(j),
              },
              {
                key: "created_at",
                header: "Created",
                render: (j) => new Date(j.created_at).toLocaleDateString(),
              },
            ]}
            rows={jobs.slice(0, 8)}
            getRowKey={(j) => j.id}
            emptyMessage="No jobs yet."
          />
        </div>
      </div>
      )}

      {activeTab === "distributed" && (
        <div className="train-layout train-layout--distributed tab-panel">
          <div className="card studio-card">
            <StudioCardHeader
              icon="D"
              title="Local distributed"
              description="Accelerate launch policy for GPUs visible to this machine"
            />
            <StudioCardBody>
              <div className="studio-checkbox-grid">
                <label className="studio-checkbox-item">
                  <input
                    type="checkbox"
                    checked={multiGpu}
                    onChange={(e) => {
                      setMultiGpu(e.target.checked);
                      if (e.target.checked && distributedStrategy === "none") {
                        setDistributedStrategy("auto");
                      }
                    }}
                    disabled={hw?.training_defaults?.multi_gpu_available === false}
                  />
                  Use local multi-GPU
                </label>
                <label className="studio-checkbox-item">
                  <input
                    type="checkbox"
                    checked={ddpFindUnused}
                    onChange={(e) => setDdpFindUnused(e.target.checked)}
                  />
                  Find unused DDP params
                </label>
              </div>
              <div className="option-grid">
                <div className="form-field">
                  <label>Strategy</label>
                  <select value={distributedStrategy} onChange={(e) => setDistributedStrategy(e.target.value)}>
                    <option value="auto">Auto</option>
                    <option value="none">Disable distributed launch</option>
                    <option value="ddp">DDP via Accelerate</option>
                  </select>
                </div>
                <div className="form-field">
                  <label>Processes per node</label>
                  <input
                    type="number"
                    min={1}
                    max={Math.max(1, LOCAL_GPU_COUNT || 8)}
                    placeholder={LOCAL_GPU_COUNT > 0 ? `Auto (${LOCAL_GPU_COUNT})` : "Auto"}
                    value={distributedNproc}
                    onChange={(e) => setDistributedNproc(e.target.value)}
                  />
                </div>
              </div>
              <div className="option-grid">
                <div className="form-field">
                  <label>Total nodes</label>
                  <input
                    type="number"
                    min={1}
                    max={256}
                    value={distributedNodes}
                    onChange={(e) => setDistributedNodes(Math.max(1, Number(e.target.value) || 1))}
                  />
                </div>
                <div className="form-field">
                  <label>This node rank</label>
                  <input
                    type="number"
                    min={0}
                    max={Math.max(0, distributedNodes - 1)}
                    value={distributedNodeRank}
                    onChange={(e) => setDistributedNodeRank(Math.max(0, Number(e.target.value) || 0))}
                  />
                </div>
              </div>
              <div className="option-grid">
                <div className="form-field">
                  <label>Master address</label>
                  <input value={distributedMasterAddr} onChange={(e) => setDistributedMasterAddr(e.target.value)} />
                </div>
                <div className="form-field">
                  <label>Master port</label>
                  <input
                    type="number"
                    min={1}
                    max={65535}
                    value={distributedMasterPort}
                    onChange={(e) => setDistributedMasterPort(Math.max(1, Number(e.target.value) || 29500))}
                  />
                </div>
              </div>
              <div className="form-field">
                <label>DDP backend</label>
                <select value={ddpBackend} onChange={(e) => setDdpBackend(e.target.value)}>
                  <option value="">Default</option>
                  <option value="nccl">NCCL</option>
                  <option value="gloo">Gloo</option>
                </select>
              </div>
              <div className="status-callout status-callout-info train-rec-panel">
                <div className="status-callout-body">
                  <strong className="status-callout-title">Launch plan</strong>
                  <div className="status-callout-text">
                    {distributedEnabled
                      ? `Accelerate DDP across ${distributedWorldSize} process${distributedWorldSize === 1 ? "" : "es"}.`
                      : "Distributed launch is disabled; the existing single-process trainer is unchanged."}
                    {LOCAL_GPU_COUNT > 0 && <> Local GPUs detected: {LOCAL_GPU_COUNT}.</>}
                  </div>
                </div>
              </div>
            </StudioCardBody>
          </div>

          <div className="card studio-card">
            <StudioCardHeader
              icon="T"
              title="Training config"
              description="Optional overrides used only when distributed launch is enabled"
            />
            <StudioCardBody>
              <label className="studio-checkbox-item studio-checkbox-item-standalone">
                <input
                  type="checkbox"
                  checked={distributedOverridesEnabled}
                  onChange={(e) => setDistributedOverridesEnabled(e.target.checked)}
                />
                Override setup settings for distributed runs
              </label>
              <div className="option-grid">
                <div className="form-field">
                  <label>Per-device batch</label>
                  <input
                    type="number"
                    min={1}
                    max={64}
                    value={distributedOverridesEnabled ? distributedBatchSize : batchSize}
                    disabled={!distributedOverridesEnabled}
                    onChange={(e) => setDistributedBatchSize(Math.max(1, Number(e.target.value) || 1))}
                  />
                </div>
                <div className="form-field">
                  <label>Grad accumulation</label>
                  <input
                    type="number"
                    min={1}
                    max={256}
                    value={distributedOverridesEnabled ? distributedGradAccum : gradAccum}
                    disabled={!distributedOverridesEnabled}
                    onChange={(e) => setDistributedGradAccum(Math.max(1, Number(e.target.value) || 1))}
                  />
                </div>
              </div>
              <div className="option-grid">
                <div className="form-field">
                  <label>Learning rate</label>
                  <input
                    type="number"
                    min={0}
                    step={0.00001}
                    value={distributedOverridesEnabled ? distributedLearningRate : lr}
                    disabled={!distributedOverridesEnabled}
                    onChange={(e) => setDistributedLearningRate(Math.max(0.000001, Number(e.target.value) || lr))}
                  />
                </div>
                <div className="form-field">
                  <label>Max seq length</label>
                  <input
                    type="number"
                    min={128}
                    step={128}
                    value={distributedOverridesEnabled ? distributedMaxSeq : maxSeq}
                    disabled={!distributedOverridesEnabled}
                    onChange={(e) => setDistributedMaxSeq(Math.max(128, Number(e.target.value) || 128))}
                  />
                </div>
              </div>
              <div className="option-grid">
                <div className="form-field">
                  <label>Epochs</label>
                  <input
                    type="number"
                    min={1}
                    max={100}
                    value={distributedOverridesEnabled ? distributedEpochs : epochs}
                    disabled={!distributedOverridesEnabled}
                    onChange={(e) => setDistributedEpochs(Math.max(1, Number(e.target.value) || 1))}
                  />
                </div>
                <div className="form-field">
                  <label>Max eval samples</label>
                  <input
                    type="number"
                    min={1}
                    max={10000}
                    value={distributedMaxEvalSamples}
                    disabled={!distributedOverridesEnabled}
                    onChange={(e) => setDistributedMaxEvalSamples(Math.max(1, Number(e.target.value) || 1))}
                  />
                </div>
              </div>
              <div className="option-grid">
                <div className="form-field">
                  <label>Logging steps</label>
                  <input
                    type="number"
                    min={1}
                    max={10000}
                    value={distributedLoggingSteps}
                    disabled={!distributedOverridesEnabled}
                    onChange={(e) => setDistributedLoggingSteps(Math.max(1, Number(e.target.value) || 1))}
                  />
                </div>
                <div className="form-field">
                  <label>Save steps</label>
                  <input
                    type="number"
                    min={1}
                    max={100000}
                    value={distributedSaveSteps}
                    disabled={!distributedOverridesEnabled}
                    onChange={(e) => setDistributedSaveSteps(Math.max(1, Number(e.target.value) || 1))}
                  />
                </div>
              </div>
              <div className="status-callout status-callout-info train-rec-panel">
                <div className="status-callout-body">
                  <strong className="status-callout-title">Effective batch</strong>
                  <div className="status-callout-text">
                    {distributedEffectiveBatch.toLocaleString()} samples per optimizer step across {distributedWorldSize} process{distributedWorldSize === 1 ? "" : "es"}.
                  </div>
                </div>
              </div>
            </StudioCardBody>
          </div>

          <div className="card studio-card studio-card-scroll">
            <StudioCardHeader
              icon="R"
              title="Run"
              description="Start with the distributed settings on this tab"
              meta={activeJob && jobStatus ? <span className={`badge badge-${jobStatus}`}>{jobStatus}</span> : undefined}
            />
            <StudioCardBody>
              <div className="status-callout status-callout-info train-rec-panel">
                <div className="status-callout-body">
                  <strong className="status-callout-title">Single-GPU fallback</strong>
                  <div className="status-callout-text">
                    Choose “Disable distributed launch” or leave local multi-GPU off to run the existing single-GPU trainer unchanged.
                  </div>
                </div>
              </div>
              <LogStream
                logs={logs}
                emptyMessage="Start a training run to stream logs here."
                fill
                label="Training log"
              />
            </StudioCardBody>
            <div className="studio-action-bar studio-action-bar-flush">
              <button
                className="btn btn-primary btn-lg"
                onClick={start}
                disabled={starting || downloadingModel || !canStart || !cloudConfigValid}
              >
                {starting ? "Starting…" : "Start training"}
              </button>
              {activeJob && (
                <button type="button" className="btn" onClick={() => setMetricsOpen(true)}>
                  View metrics
                </button>
              )}
            </div>
          </div>
        </div>
      )}

      {activeTab === "cloud" && (
        <div className="train-layout train-layout--distributed tab-panel">
          <div className="card studio-card">
            <StudioCardHeader
              icon="C"
              title="Cloud target"
              description="Provider, region, and GPU capacity for distributed launchers"
            />
            <StudioCardBody>
              <label className="studio-checkbox-item studio-checkbox-item-standalone">
                <input
                  type="checkbox"
                  checked={cloudGpuEnabled}
                  onChange={(e) => {
                    setCloudGpuEnabled(e.target.checked);
                    if (!e.target.checked) {
                      setCloudGpuProvider("none");
                      setSelectedCloudCredentialId("");
                    }
                  }}
                />
                Use cloud GPUs for distributed training
              </label>
              <div className="option-grid">
                <div className="form-field">
                  <label>Provider</label>
                  <select
                    value={cloudGpuProvider}
                    onChange={(e) => setCloudGpuProvider(e.target.value)}
                    disabled={!cloudGpuEnabled}
                  >
                    <option value="none">None</option>
                    <option value="aws">AWS</option>
                    <option value="gcp">Google Cloud</option>
                    <option value="azure">Azure</option>
                    <option value="lambda">Lambda</option>
                    <option value="runpod">RunPod</option>
                    <option value="coreweave">CoreWeave</option>
                    <option value="custom">Custom scheduler</option>
                  </select>
                </div>
                <div className="form-field">
                  <label>GPU count</label>
                  <input
                    type="number"
                    min={1}
                    max={1024}
                    value={cloudGpuCount}
                    disabled={!cloudGpuEnabled}
                    onChange={(e) => setCloudGpuCount(Math.max(1, Number(e.target.value) || 1))}
                  />
                </div>
              </div>
              <div className="option-grid">
                <div className="form-field">
                  <label>Region / zone</label>
                  <input
                    value={cloudGpuRegion}
                    disabled={!cloudGpuEnabled}
                    onChange={(e) => setCloudGpuRegion(e.target.value)}
                    placeholder="us-east-1 or us-central1-a"
                  />
                </div>
                <div className="form-field">
                  <label>Instance type</label>
                  <input
                    value={cloudGpuInstanceType}
                    disabled={!cloudGpuEnabled}
                    onChange={(e) => setCloudGpuInstanceType(e.target.value)}
                    placeholder="p5.48xlarge, a3-highgpu, etc."
                  />
                </div>
              </div>
              <div className="form-field">
                <label>Project label</label>
                <input
                  value={cloudGpuProject}
                  disabled={!cloudGpuEnabled}
                  onChange={(e) => setCloudGpuProject(e.target.value)}
                  placeholder="Optional billing or scheduler label"
                />
              </div>
              <div className="form-field">
                <label>Saved cloud access</label>
                <select
                  value={selectedCloudCredentialId}
                  disabled={!cloudGpuEnabled}
                  onChange={(e) => setSelectedCloudCredentialId(e.target.value)}
                >
                  <option value="">Select encrypted credential</option>
                  {cloudCredentials.map((cred) => (
                    <option key={cred.id} value={cred.id}>
                      {cred.name} ({cred.config.provider || "cloud"})
                    </option>
                  ))}
                </select>
              </div>
              <div className={`status-callout ${cloudConfigValid ? "status-callout-info" : "status-callout-error"} train-rec-panel`}>
                <div className="status-callout-body">
                  <strong className="status-callout-title">Cloud launch readiness</strong>
                  <div className="status-callout-text">
                    Cloud distributed jobs reference a saved encrypted credential by id. Secrets are not copied into the training job config.
                  </div>
                </div>
              </div>
            </StudioCardBody>
          </div>

          <div className="card studio-card">
            <StudioCardHeader
              icon="K"
              title="Add cloud access"
              description="API keys, provider tokens, SSH material, and bootstrap commands"
            />
            <StudioCardBody>
              <div className="option-grid">
                <div className="form-field">
                  <label>Name</label>
                  <input
                    value={cloudCredentialName}
                    onChange={(e) => setCloudCredentialName(e.target.value)}
                    placeholder="prod-a100-pool"
                  />
                </div>
                <div className="form-field">
                  <label>Auth type</label>
                  <select value={cloudAuthKind} onChange={(e) => setCloudAuthKind(e.target.value)}>
                    <option value="api_key">API key / provider token</option>
                    <option value="aws_keys">AWS access keys</option>
                    <option value="ssh">SSH key</option>
                    <option value="scheduler">Custom scheduler</option>
                  </select>
                </div>
              </div>
              <div className="form-field">
                <label>API key / provider token</label>
                <input
                  type="password"
                  value={cloudApiKey}
                  onChange={(e) => setCloudApiKey(e.target.value)}
                  autoComplete="off"
                  placeholder="Stored encrypted locally"
                />
              </div>
              <div className="option-grid">
                <div className="form-field">
                  <label>Access key id</label>
                  <input
                    type="password"
                    value={cloudAccessKeyId}
                    onChange={(e) => setCloudAccessKeyId(e.target.value)}
                    autoComplete="off"
                  />
                </div>
                <div className="form-field">
                  <label>Secret access key</label>
                  <input
                    type="password"
                    value={cloudSecretAccessKey}
                    onChange={(e) => setCloudSecretAccessKey(e.target.value)}
                    autoComplete="off"
                  />
                </div>
              </div>
              <div className="form-field">
                <label>Session token</label>
                <textarea
                  value={cloudSessionToken}
                  onChange={(e) => setCloudSessionToken(e.target.value)}
                  rows={3}
                  placeholder="Optional temporary provider session token"
                />
              </div>
              <div className="form-field">
                <label>SSH username</label>
                <input
                  value={cloudSshUsername}
                  onChange={(e) => setCloudSshUsername(e.target.value)}
                  placeholder="ubuntu, ec2-user, etc."
                />
              </div>
              <div className="form-field">
                <label>SSH private key</label>
                <textarea
                  value={cloudSshPrivateKey}
                  onChange={(e) => setCloudSshPrivateKey(e.target.value)}
                  rows={5}
                  placeholder="Stored encrypted locally; never shown after save"
                />
              </div>
              <div className="form-field">
                <label>Bootstrap command</label>
                <textarea
                  value={cloudBootstrapCommand}
                  onChange={(e) => setCloudBootstrapCommand(e.target.value)}
                  rows={4}
                  placeholder="Optional scheduler-side setup command"
                />
              </div>
            </StudioCardBody>
            <div className="studio-action-bar studio-action-bar-flush">
              <button
                type="button"
                className="btn btn-primary"
                onClick={saveCloudCredential}
                disabled={!cloudCredentialName.trim() || cloudGpuProvider === "none"}
              >
                Save encrypted access
              </button>
            </div>
          </div>

          <div className="card studio-card studio-card-scroll">
            <StudioCardHeader
              icon="S"
              title="Saved access"
              description="Encrypted credentials available to distributed launchers"
              meta={<span className="badge badge-dim">{cloudCredentials.length}</span>}
            />
            <StudioCardBody>
              {cloudCredentialMsg && (
                <div className="status-callout status-callout-info train-rec-panel">
                  <div className="status-callout-body">
                    <div className="status-callout-text">{cloudCredentialMsg}</div>
                  </div>
                </div>
              )}
              {cloudCredentials.length === 0 ? (
                <p className="muted-text">No cloud access credentials saved yet.</p>
              ) : (
                <div className="studio-monitor-stack">
                  {cloudCredentials.map((cred) => (
                    <div key={cred.id} className="status-callout status-callout-info train-rec-panel">
                      <div className="status-callout-body">
                        <strong className="status-callout-title">{cred.name}</strong>
                        <div className="status-callout-text">
                          {cred.config.provider || "cloud"} · {cred.config.region || "region unset"} ·{" "}
                          {cred.config.auth_kind || "auth"}
                        </div>
                        <div className="muted-text studio-field-hint studio-field-hint-compact">
                          API key: {cred.config.api_key_configured ? "saved" : "none"} · SSH key:{" "}
                          {cred.config.ssh_private_key_configured ? "saved" : "none"} · Bootstrap:{" "}
                          {cred.config.bootstrap_command_configured ? "saved" : "none"}
                        </div>
                        <div className="form-actions">
                          <button
                            type="button"
                            className="btn btn-sm"
                            onClick={() => setSelectedCloudCredentialId(cred.id)}
                          >
                            Use for training
                          </button>
                          <button
                            type="button"
                            className="btn btn-sm"
                            onClick={() => deleteCloudCredential(cred.id)}
                          >
                            Delete
                          </button>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </StudioCardBody>
          </div>
        </div>
      )}
    </StudioPageShell>
  );
}
