import { useCallback, useEffect, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { api, subscribeSSE, SystemMetrics, TrainingJob, TrainingMetricPoint, TrainingRecommendations } from "@/lib/api";
import { invalidateApiCache } from "@/lib/api/getCache";
import { appendBoundedLog } from "@/lib/api/sse";
import { initialDownloadProgress, ModelProgressState } from "@/lib/modelProgress";
import { ensureTrainHubModel, isTrainModelCached } from "@/lib/trainModel";
import { GGUF_TRAIN_ERROR, isGgufOnlyRepoId } from "@/lib/trainRepo";
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
import { TrainingMetricsDashboard } from "@/components/TrainingMetricsDashboard";
import { useHardwareProfile } from "@/hooks/useHardware";

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
  const [dataset, setDataset] = useState("HuggingFaceH4/no_robots");
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
  const [validatingDataset, setValidatingDataset] = useState(false);

  const validateDataset = useCallback(async (ds: string, fmt: string) => {
    if (!ds) {
      setDatasetValid(true);
      setDatasetError(null);
      return;
    }
    setValidatingDataset(true);
    try {
      const res = await api.validateDataset(ds, fmt);
      if (res.valid) {
        setDatasetValid(true);
        setDatasetError(null);
      } else {
        setDatasetValid(false);
        setDatasetError(res.error || "Dataset cannot be normalized for training");
      }
    } catch (e: any) {
      setDatasetValid(false);
      setDatasetError(e?.message || "Failed to validate dataset");
    } finally {
      setValidatingDataset(false);
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
    if (pendingModel) setModelId(pendingModel);
  }, [pendingModel]);

  useEffect(() => {
    if (pendingModel || modelId) return;
    const stored = readStoredModel("train:model");
    if (stored) {
      setModelId(stored);
      return;
    }
    const firstLocal = localModels.find((m) => m.repo_id)?.repo_id;
    if (firstLocal) setModelId(firstLocal);
  }, [localModels, pendingModel, modelId]);

  useEffect(() => {
    if (!pendingModel || !pendingDownload) return;
    if (isGgufOnlyRepoId(pendingModel)) {
      setDownloadError(GGUF_TRAIN_ERROR);
      setSearchParams({ model: pendingModel }, { replace: true });
      return;
    }

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
    if (hw.recommended_train_repo && !readStoredModel("train:model") && !modelId) {
      setModelId(hw.recommended_train_repo);
    }
    setHwApplied(true);
  }, [hw, hwApplied, pendingModel]);

  useEffect(() => {
    if (method === "full") setExportProfile("full_bundle");
    else if (method === "lora") setExportProfile("lora_bundle");
  }, [method]);

  useEffect(() => {
    if (!modelId && !dataset) {
      setRecommendations(null);
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

  // Pre-validate dataset for format/normalization before allowing training start
  useEffect(() => {
    const t = setTimeout(() => {
      validateDataset(dataset, datasetFormat);
    }, 250);
    return () => clearTimeout(t);
  }, [dataset, datasetFormat, validateDataset]);

  useEffect(() => {
    if (configCustomized || !recommendations?.config) return;
    const rec = recommendations.config;
    if (rec.dataset_format && rec.dataset_format !== "auto") {
      setDatasetFormat(rec.dataset_format);
    }
    if (rec.train_on_responses_only != null) {
      setTrainResponsesOnly(rec.train_on_responses_only);
    }
  }, [recommendations, configCustomized]);

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

  const modelBlocked = Boolean(modelId && (isGgufOnlyRepoId(modelId) || recommendations?.trainable === false));

  useEffect(() => {
    if (!exportProfiles.length) return;
    const prof = exportProfiles.find((p) => p.id === exportProfile);
    if (prof?.default_gguf_quants?.length) {
      setExportQuants(prof.default_gguf_quants);
    }
  }, [exportProfile, exportProfiles]);

  const toggleExportQuant = (quant: string) => {
    setExportQuants((prev) =>
      prev.includes(quant) ? prev.filter((q) => q !== quant) : [...prev, quant],
    );
  };

  const GGUF_QUANT_OPTIONS = ["q2_k", "q3_k_m", "q4_k_m", "q5_k_m", "q6_k", "q8_0", "f16"];

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

  const start = async () => {
    if (modelBlocked) {
      setDownloadError(recommendations?.warnings[0] || GGUF_TRAIN_ERROR);
      return;
    }
    if (!datasetValid) {
      setDownloadError(datasetError || "Dataset cannot be normalized for training");
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

      const res = await api.startTraining(
        {
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
        },
        multiGpu,
        exportPayload,
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
      subtitle="Fine-tune with LoRA/QLoRA — pick a safetensors base model (not GGUF). Settings adapt to your GPU and dataset."
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
            <strong className="status-callout-title">Download failed</strong>
            <div className="status-callout-text">{downloadError}</div>
          </div>
        </div>
      )}

      <div className="train-layout train-layout--studio train-layout--studio-fit">
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
                  {recommendations?.warnings[0] || GGUF_TRAIN_ERROR}
                  {recommendations?.fallback_train_repo && (
                    <>
                      {" "}
                      Try{" "}
                      <button
                        type="button"
                        className="btn-link"
                        onClick={() => {
                          setModelId(recommendations.fallback_train_repo!);
                          writeStoredModel("train:model", recommendations.fallback_train_repo!);
                        }}
                      >
                        {recommendations.fallback_train_repo}
                      </button>
                      .
                    </>
                  )}
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

          {(validatingDataset || !datasetValid || datasetError) && (
            <div className={`status-callout ${datasetValid ? "status-callout-warn" : "status-callout-error"} studio-error-callout`} style={{marginTop: 8}}>
              <div className="status-callout-body">
                <strong className="status-callout-title">
                  {validatingDataset ? "Validating dataset…" : datasetValid ? "Dataset warning" : "Dataset error"}
                </strong>
                <div className="status-callout-text">
                  {validatingDataset ? "Checking if the dataset can be normalized for training..." : (datasetError || "Dataset looks invalid for training.")}
                </div>
                {!datasetValid && (
                  <div className="muted-text" style={{fontSize: "12px", marginTop: 4}}>
                    Fix the format or pick a dataset with chat messages, preference pairs (chosen/rejected), instruction/output pairs, or plain text.
                  </div>
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
                <input
                  type="checkbox"
                  checked={multiGpu}
                  onChange={(e) => setMultiGpu(e.target.checked)}
                  disabled={hw?.training_defaults?.multi_gpu_available === false}
                />
                Multi-GPU
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
              disabled={starting || downloadingModel || modelBlocked || !datasetValid || validatingDataset}
            >
              {starting ? "Starting…" : downloadingModel ? "Downloading model…" : "Start training"}
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
    </StudioPageShell>
  );
}
