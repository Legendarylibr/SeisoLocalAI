import { useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { api, CatalogModel, subscribeSSE, SystemMetrics, TrainableModel, TrainingJob, TrainingMetricPoint } from "@/lib/api";
import { HfDatasetPicker } from "@/components/HfDatasetPicker";
import { TrainingMetricsDashboard } from "@/components/TrainingMetricsDashboard";
import { useHardwareProfile } from "@/hooks/useHardware";

export function TrainPage() {
  const { profile: hw } = useHardwareProfile();
  const [searchParams] = useSearchParams();
  const pendingModel = searchParams.get("model");
  const [jobs, setJobs] = useState<TrainingJob[]>([]);
  const [catalog, setCatalog] = useState<CatalogModel[]>([]);
  const [localModels, setLocalModels] = useState<TrainableModel[]>([]);
  const [modelId, setModelId] = useState("meta-llama/Llama-3.2-1B-Instruct");
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
  const [logs, setLogs] = useState<string[]>([]);
  const [activeJob, setActiveJob] = useState<string | null>(null);
  const [jobStatus, setJobStatus] = useState<string | null>(null);
  const [trainingMetrics, setTrainingMetrics] = useState<TrainingMetricPoint[]>([]);
  const [systemMetrics, setSystemMetrics] = useState<SystemMetrics[]>([]);
  const [metricsOpen, setMetricsOpen] = useState(false);
  const [starting, setStarting] = useState(false);
  const [hwApplied, setHwApplied] = useState(false);

  useEffect(() => {
    api.catalog("", undefined, undefined).then((r) => setCatalog(r.models)).catch(console.error);
    api.listTrainingJobs().then(setJobs).catch(console.error);
    api.listTrainingModels().then((r) => setLocalModels(r.models)).catch(console.error);
    if (pendingModel) setModelId(pendingModel);
  }, [pendingModel]);

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
    if (hw.recommended_train_repo) setModelId(hw.recommended_train_repo);
    setHwApplied(true);
  }, [hw, hwApplied, pendingModel]);

  const start = async () => {
    setStarting(true);
    setLogs([]);
    setTrainingMetrics([]);
    setSystemMetrics([]);
    setJobStatus("running");
    try {
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
      );
      setActiveJob(res.job_id);
      setMetricsOpen(true);
      subscribeSSE(`/training/jobs/${res.job_id}/stream`, (event, data) => {
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
      });
      api.listTrainingJobs().then(setJobs);
    } finally {
      setStarting(false);
    }
  };

  const chatModels = catalog.filter((m) => m.task === "chat" || m.task === "code");
  const cachedRepoIds = new Set(localModels.map((m) => m.repo_id).filter(Boolean) as string[]);

  return (
    <div className="train-page">
      <TrainingMetricsDashboard
        jobId={activeJob}
        open={metricsOpen}
        onClose={() => setMetricsOpen(false)}
        trainingPoints={trainingMetrics}
        systemPoints={systemMetrics}
        status={jobStatus}
      />

      <h1 className="page-title">Training Studio</h1>
      <p className="page-sub">
        QLoRA 4-bit, TRL SFTTrainer — settings below are pre-filled from local hardware detection (nothing leaves this machine).
      </p>
      {hw?.training_defaults && (
        <div className="hw-inline-banner card">
          <span className="trust-badge">{hw.tier_label}</span>
          <span className="muted-text">{hw.training_defaults.note}</span>
        </div>
      )}

      <div className="train-layout">
        <div className="card">
          <h3 className="section-title">Model & data</h3>
          <label>Base model (HF repo ID)</label>
          <input list="train-models" value={modelId} onChange={(e) => setModelId(e.target.value)} />
          <datalist id="train-models">
            {localModels.map((m) => (
              <option key={m.id} value={m.repo_id || m.path}>
                {m.name} (cached locally)
              </option>
            ))}
            {chatModels.map((m) => (
              <option key={m.repo_id} value={m.repo_id}>
                {m.name}{cachedRepoIds.has(m.repo_id) ? " · cached" : ""}
              </option>
            ))}
          </datalist>
          {localModels.length > 0 && (
            <p className="muted-text" style={{ marginTop: "0.35rem" }}>
              {localModels.length} safetensors snapshot{localModels.length === 1 ? "" : "s"} ready on disk — training uses cached weights automatically.
            </p>
          )}
          <label>Dataset</label>
          <HfDatasetPicker value={dataset} onChange={setDataset} />
          <label>Dataset format</label>
          <select value={datasetFormat} onChange={(e) => setDatasetFormat(e.target.value)}>
            <option value="auto">Auto-detect</option>
            <option value="chat">Chat / messages</option>
            <option value="alpaca">Alpaca (instruction/output)</option>
            <option value="sharegpt">ShareGPT conversations</option>
            <option value="text">Plain text</option>
          </select>
        </div>

        <div className="card">
          <h3 className="section-title">Training method</h3>
          <div className="option-grid">
            <label>Method</label>
            <select value={method} onChange={(e) => setMethod(e.target.value)}>
              <option value="lora">LoRA / QLoRA</option>
              <option value="full">Full fine-tune</option>
              <option value="embedding">Embedding</option>
            </select>
            <label>Quantization</label>
            <select value={quant} onChange={(e) => setQuant(e.target.value)}>
              <option value="4bit">4-bit (QLoRA)</option>
              <option value="8bit">8-bit</option>
              <option value="16bit">16-bit FP16</option>
              <option value="none">None (full precision)</option>
            </select>
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
          {method === "lora" && (
            <>
              <div className="option-grid">
                <div>
                  <label>LoRA rank (r): {loraR}</label>
                  <input type="range" min={4} max={128} step={4} value={loraR} onChange={(e) => setLoraR(+e.target.value)} />
                </div>
                <div>
                  <label>LoRA alpha: {loraAlpha}</label>
                  <input type="range" min={8} max={256} step={8} value={loraAlpha} onChange={(e) => setLoraAlpha(+e.target.value)} />
                </div>
              </div>
              <label>Grad accumulation: {gradAccum}</label>
              <input type="range" min={1} max={32} value={gradAccum} onChange={(e) => setGradAccum(+e.target.value)} />
            </>
          )}
          <div className="checkbox-group">
            <label><input type="checkbox" checked={gradCkpt} onChange={(e) => setGradCkpt(e.target.checked)} /> Gradient checkpointing</label>
            <label><input type="checkbox" checked={trainResponsesOnly} onChange={(e) => setTrainResponsesOnly(e.target.checked)} /> Train on responses only</label>
            <label><input type="checkbox" checked={useRsLora} onChange={(e) => setUseRsLora(e.target.checked)} /> Rank-stabilized LoRA (rsLoRA)</label>
            <label><input type="checkbox" checked={packing} onChange={(e) => setPacking(e.target.checked)} /> Sequence packing</label>
            <label>
              <input
                type="checkbox"
                checked={multiGpu}
                onChange={(e) => setMultiGpu(e.target.checked)}
                disabled={hw?.training_defaults?.multi_gpu_available === false}
              />
              {" "}Multi-GPU
            </label>
            <label title={hw?.training_defaults?.kernel_backend ? `Backend: ${hw.training_defaults.kernel_backend}` : undefined}>
              <input
                type="checkbox"
                checked={useFusedKernels}
                onChange={(e) => setUseFusedKernels(e.target.checked)}
                disabled={hw?.training_defaults?.use_fused_kernels === false}
              />
              {" "}Fused kernels (RMSNorm + SwiGLU)
            </label>
            <label>
              <input
                type="checkbox"
                checked={useFusedCe}
                onChange={(e) => setUseFusedCe(e.target.checked)}
                disabled={hw?.training_defaults?.use_fused_ce === false}
              />
              {" "}Fused cross-entropy
            </label>
          </div>
          <button className="btn btn-primary btn-lg" onClick={start} disabled={starting}>
            {starting ? "Starting…" : "Start training"}
          </button>
          {activeJob && (
            <button
              type="button"
              className="btn"
              style={{ marginLeft: "0.75rem", marginTop: "0.5rem", width: "auto" }}
              onClick={() => setMetricsOpen(true)}
            >
              Open metrics dashboard
            </button>
          )}
        </div>
      </div>

      {logs.length > 0 && (
        <div className="card">
          <h3 className="section-title">
            Live log {activeJob && <span className="badge">{activeJob.slice(0, 8)}</span>}
          </h3>
          <div className="log-panel log-panel-tall">{logs.join("\n")}</div>
        </div>
      )}

      <div className="card">
        <h3 className="section-title">Training history</h3>
        {jobs.length === 0 ? (
          <p className="muted-text">No training jobs yet.</p>
        ) : (
          <table>
            <thead><tr><th>Job</th><th>Status</th><th>Created</th></tr></thead>
            <tbody>
              {jobs.map((j) => (
                <tr key={j.id}>
                  <td className="mono">{j.id.slice(0, 8)}…</td>
                  <td><span className={`badge badge-${j.status}`}>{j.status}</span></td>
                  <td className="muted-cell">{new Date(j.created_at).toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
