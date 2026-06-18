import { useEffect, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { api, subscribeSSE, SystemMetrics, TrainingJob, TrainingMetricPoint } from "@/lib/api";
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
  const [epochs, setEpochs] = useState(1);
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
      await refreshLocalModels();
    } finally {
      setDownloadingModel(false);
      setLoadProgress(null);
    }
  };

  const start = async () => {
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
          if (event === "log") setLogs((l) => [...l, data]);
          if (event === "error") setLogs((l) => [...l, `ERROR: ${data}`]);
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
            api.listTrainingJobs().then(setJobs);
          }
        },
        (err) => setLogs((l) => [...l, `ERROR: ${err.message}`]),
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
      subtitle="QLoRA 4-bit, TRL SFTTrainer — settings below are pre-filled from local hardware detection (nothing leaves this machine)."
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
          <div className="studio-card-head">
            <span className="studio-card-icon" aria-hidden>↓</span>
            <div className="studio-card-head-text">
              <div className="studio-card-title">Downloading base model</div>
              <div className="studio-card-desc">Fetching safetensors snapshot from Hugging Face Hub</div>
            </div>
          </div>
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

      <div className="train-layout">
        <div className="card studio-card">
          <div className="studio-card-head">
            <span className="studio-card-icon" aria-hidden>①</span>
            <div className="studio-card-head-text">
              <div className="studio-card-title">Model & data</div>
              <div className="studio-card-desc">Base checkpoint and training dataset</div>
            </div>
          </div>
          <div className="form-field">
            <label>Base model</label>
            <HfBaseModelPicker
              value={modelId}
              localModels={localModels}
              disabled={downloadingModel}
              onChange={(value) => {
                setModelId(value);
                writeStoredModel("train:model", value);
              }}
            />
          </div>
          {localModels.length > 0 && (
            <p className="muted-text studio-field-hint">
              {localModels.length} safetensors snapshot{localModels.length === 1 ? "" : "s"} ready on disk — training uses cached weights automatically.
            </p>
          )}
          <div className="form-field">
            <label>Dataset</label>
            <HfDatasetPicker value={dataset} onChange={setDataset} />
          </div>
          <div className="form-field">
            <label>Dataset format</label>
            <select value={datasetFormat} onChange={(e) => setDatasetFormat(e.target.value)}>
              <option value="auto">Auto-detect</option>
              <option value="chat">Chat / messages</option>
              <option value="alpaca">Alpaca (instruction/output)</option>
              <option value="sharegpt">ShareGPT conversations</option>
              <option value="text">Plain text</option>
            </select>
          </div>
        </div>

        <div className="card studio-card">
          <div className="studio-card-head">
            <span className="studio-card-icon" aria-hidden>②</span>
            <div className="studio-card-head-text">
              <div className="studio-card-title">Training method</div>
              <div className="studio-card-desc">Hyperparameters, LoRA, and optimization flags</div>
            </div>
          </div>

          <FormSection title="Hyperparameters" hint="Method, quantization, and core training knobs.">
            <div className="option-grid">
              <div className="form-field">
                <label>Method</label>
                <select value={method} onChange={(e) => setMethod(e.target.value)}>
                  <option value="lora">LoRA / QLoRA</option>
                  <option value="full">Full fine-tune</option>
                  <option value="embedding">Embedding</option>
                </select>
              </div>
              <div className="form-field">
                <label>Quantization</label>
                <select value={quant} onChange={(e) => setQuant(e.target.value)}>
                  <option value="4bit">4-bit (QLoRA)</option>
                  <option value="8bit">8-bit</option>
                  <option value="16bit">16-bit FP16</option>
                  <option value="none">None (full precision)</option>
                </select>
              </div>
            </div>
            <div className="slider-row">
              <label>Epochs: {epochs}</label>
              <input type="range" min={1} max={10} value={epochs} onChange={(e) => setEpochs(+e.target.value)} />
            </div>
            <div className="slider-row">
              <label>Batch size: {batchSize}</label>
              <input type="range" min={1} max={16} value={batchSize} onChange={(e) => setBatchSize(+e.target.value)} />
            </div>
            <div className="slider-row">
              <label>Max seq length: {maxSeq}</label>
              <input type="range" min={512} max={8192} step={256} value={maxSeq} onChange={(e) => setMaxSeq(+e.target.value)} />
            </div>
            <div className="slider-row">
              <label>Learning rate: {lr.toExponential(1)}</label>
              <input type="range" min={-6} max={-3} step={0.1} value={Math.log10(lr)} onChange={(e) => setLr(10 ** +e.target.value)} />
            </div>
          </FormSection>

          {method === "lora" && (
            <FormSection title="LoRA settings" hint="Rank, alpha, and gradient accumulation.">
              <div className="option-grid">
                <div className="form-field">
                  <label>LoRA rank (r): {loraR}</label>
                  <input type="range" min={4} max={128} step={4} value={loraR} onChange={(e) => setLoraR(+e.target.value)} />
                </div>
                <div className="form-field">
                  <label>LoRA alpha: {loraAlpha}</label>
                  <input type="range" min={8} max={256} step={8} value={loraAlpha} onChange={(e) => setLoraAlpha(+e.target.value)} />
                </div>
              </div>
              <div className="slider-row">
                <label>Grad accumulation: {gradAccum}</label>
                <input type="range" min={1} max={32} value={gradAccum} onChange={(e) => setGradAccum(+e.target.value)} />
              </div>
            </FormSection>
          )}

          <FormSection title="Optimization" hint="Memory and throughput trade-offs." collapsible defaultOpen>
            <div className="studio-checkbox-grid">
              <label className="studio-checkbox-item">
                <input type="checkbox" checked={gradCkpt} onChange={(e) => setGradCkpt(e.target.checked)} />
                Gradient checkpointing
              </label>
              <label className="studio-checkbox-item">
                <input type="checkbox" checked={trainResponsesOnly} onChange={(e) => setTrainResponsesOnly(e.target.checked)} />
                Train on responses only
              </label>
              <label className="studio-checkbox-item">
                <input type="checkbox" checked={useRsLora} onChange={(e) => setUseRsLora(e.target.checked)} />
                Rank-stabilized LoRA (rsLoRA)
              </label>
              <label className="studio-checkbox-item">
                <input type="checkbox" checked={packing} onChange={(e) => setPacking(e.target.checked)} />
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
          </FormSection>

          {method !== "embedding" && (
            <FormSection
              title="Post-training export"
              hint="Bundle LoRA, merged weights, and GGUF quants when the run finishes."
              collapsible
              defaultOpen={exportOnComplete}
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
        </div>
      </div>

      <div className="card studio-card studio-run-card">
        <div className="studio-card-head studio-card-head-inline">
          <span className="studio-card-icon" aria-hidden>▶</span>
          <div className="studio-card-head-text">
            <div className="studio-card-title">Run training</div>
            <div className="studio-card-desc">
              Downloads the base model if needed, then starts a local QLoRA job on this machine.
            </div>
            {activeJob && jobStatus && (
              <div className="studio-card-meta">
                <span className={`badge badge-${jobStatus}`}>{jobStatus}</span>
              </div>
            )}
          </div>
        </div>
        <div className="studio-action-bar studio-action-bar-flush">
          <button className="btn btn-primary btn-lg" onClick={start} disabled={starting || downloadingModel}>
            {starting ? "Starting…" : downloadingModel ? "Downloading model…" : "Start training"}
          </button>
          {activeJob && (
            <button type="button" className="btn" onClick={() => setMetricsOpen(true)}>
              View metrics
            </button>
          )}
        </div>
      </div>

      <div className="train-layout studio-monitor-layout">
        <div className="card studio-card">
          <div className="studio-card-head">
            <span className="studio-card-icon" aria-hidden>③</span>
            <div className="studio-card-head-text">
              <div className="studio-card-title">Live output</div>
              <div className="studio-card-desc">Streaming logs from the active training job</div>
              {activeJob && (
                <div className="studio-card-meta">
                  <span className="mono studio-job-id">{activeJob.slice(0, 8)}…</span>
                </div>
              )}
            </div>
          </div>
          <LogStream
            logs={logs}
            emptyMessage="Start a training run to stream logs here."
            tall
          />
        </div>

        <div className="card studio-card">
          <div className="studio-card-head">
            <span className="studio-card-icon" aria-hidden>④</span>
            <div className="studio-card-head-text">
              <div className="studio-card-title">Training history</div>
              <div className="studio-card-desc">Past jobs on this machine</div>
              {jobs.length > 0 && (
                <div className="studio-card-meta">
                  <span className="badge badge-dim">{jobs.length} job{jobs.length === 1 ? "" : "s"}</span>
                </div>
              )}
            </div>
          </div>
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
                render: (j) => new Date(j.created_at).toLocaleString(),
              },
            ]}
            rows={jobs}
            getRowKey={(j) => j.id}
            emptyMessage="No training jobs yet — configure settings above and start your first run."
          />
        </div>
      </div>
    </StudioPageShell>
  );
}
