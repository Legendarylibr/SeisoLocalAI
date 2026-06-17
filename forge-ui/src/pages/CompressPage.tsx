import { useEffect, useState } from "react";
import { api, CompressJob, CompressPreset, subscribeSSE } from "@/lib/api";
import { StudioPageShell } from "@/components/StudioPageShell";

const FALLBACK_PRESETS: CompressPreset[] = [
  { id: "smoke", label: "Smoke", stages: ["distill", "prune", "finetune", "evaluate", "export"] },
  { id: "full", label: "Full", stages: ["distill", "prune", "finetune", "evaluate", "export"] },
  { id: "distill_only", label: "Distill Only", stages: ["distill", "evaluate"] },
  { id: "prune_recover", label: "Prune Recover", stages: ["prune", "finetune", "evaluate", "export"] },
  { id: "quantize", label: "Quantize", stages: ["quantize_gptq", "evaluate", "export"] },
];

const FALLBACK_STAGES = [
  "distill",
  "prune",
  "finetune",
  "evaluate",
  "export",
  "quantize_gptq",
  "quantize_awq",
];

export function CompressPage() {
  const [jobs, setJobs] = useState<CompressJob[]>([]);
  const [presets, setPresets] = useState<CompressPreset[]>([]);
  const [allStages, setAllStages] = useState<string[]>(FALLBACK_STAGES);
  const [stageHelp, setStageHelp] = useState<Record<string, string>>({});
  const [preset, setPreset] = useState("smoke");
  const [selectedStages, setSelectedStages] = useState<string[]>(FALLBACK_PRESETS[0].stages);
  const [teacherModel, setTeacherModel] = useState("codellama/CodeLlama-13b-hf");
  const [studentModel, setStudentModel] = useState("codellama/CodeLlama-7b-hf");
  const [modelDir, setModelDir] = useState("");
  const [distillSteps, setDistillSteps] = useState<number | "">("");
  const [finetuneSteps, setFinetuneSteps] = useState<number | "">("");
  const [pruneRatio, setPruneRatio] = useState(0.25);
  const [pruneMethod, setPruneMethod] = useState("magnitude");
  const [maxTrainSamples, setMaxTrainSamples] = useState<number | "">("");
  const [calibrationSamples, setCalibrationSamples] = useState<number | "">("");
  const [exportModelName, setExportModelName] = useState("seiso-compressed");
  const [seed, setSeed] = useState(42);
  const [deterministic, setDeterministic] = useState(true);
  const [configFile, setConfigFile] = useState("");
  const [linkTrainingJob, setLinkTrainingJob] = useState("");
  const [logs, setLogs] = useState<string[]>([]);
  const [result, setResult] = useState<Record<string, unknown> | null>(null);
  const [activeJob, setActiveJob] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);

  const presetList = presets.length ? presets : FALLBACK_PRESETS;

  useEffect(() => {
    api.listCompressJobs().then(setJobs).catch(console.error);
    api.compressPresets().then((r) => {
      setPresets(r.presets);
      setAllStages(r.stages.length ? r.stages : FALLBACK_STAGES);
      setStageHelp(r.help);
    }).catch(console.error);
  }, []);

  useEffect(() => {
    const p = (presets.length ? presets : FALLBACK_PRESETS).find((x) => x.id === preset);
    if (p?.stages.length) setSelectedStages(p.stages);
  }, [preset, presets]);

  const refreshJobs = () => api.listCompressJobs().then(setJobs).catch(console.error);

  const toggleStage = (stage: string) => {
    setSelectedStages((prev) =>
      prev.includes(stage) ? prev.filter((s) => s !== stage) : [...prev, stage],
    );
  };

  const start = async () => {
    setStarting(true);
    setLogs([]);
    setResult(null);
    try {
      const body: Record<string, unknown> = {
        preset,
        teacher_model: teacherModel,
        student_model: studentModel,
        prune_ratio: pruneRatio,
        prune_method: pruneMethod,
        seed,
        deterministic,
        export_model_name: exportModelName,
      };
      if (selectedStages.length) body.stages = selectedStages;
      if (modelDir) body.model_dir = modelDir;
      if (configFile) body.config_file = configFile;
      if (distillSteps !== "") body.distill_steps = distillSteps;
      if (finetuneSteps !== "") body.finetune_steps = finetuneSteps;
      if (maxTrainSamples !== "") body.max_train_samples = maxTrainSamples;
      if (calibrationSamples !== "") body.calibration_samples = calibrationSamples;
      if (linkTrainingJob) body.link_training_job_id = linkTrainingJob;

      const res = await api.startCompress(body);
      setActiveJob(res.job_id);
      subscribeSSE(`/compress/jobs/${res.job_id}/stream`, (event, data) => {
        if (event === "log") setLogs((l) => [...l, data]);
        if (event === "error") setLogs((l) => [...l, `ERROR: ${data}`]);
        if (event === "result") {
          try {
            setResult(JSON.parse(data));
          } catch {
            /* ignore */
          }
          refreshJobs();
        }
      });
      refreshJobs();
    } finally {
      setStarting(false);
    }
  };

  return (
    <StudioPageShell
      title="Model Compression"
      subtitle="Code Llama compression pipeline: distillation, MLP pruning, recovery fine-tune, evaluation, and export bundles (vLLM/Docker/GGUF scripts). Hash-chained manifests for reproducibility."
    >
      <div className="train-layout">
        <div className="card compress-config-card studio-card">
          <div className="studio-card-head">
            <span className="studio-card-icon" aria-hidden>⚙</span>
            <div className="studio-card-head-text">
              <div className="studio-card-title">Pipeline configuration</div>
              <div className="studio-card-desc">Presets, stages, models, and training parameters</div>
            </div>
          </div>
          <h3 className="section-title">Pipeline</h3>
          <label>Preset</label>
          <select value={preset} onChange={(e) => setPreset(e.target.value)}>
            {presetList.map((p) => (
              <option key={p.id} value={p.id}>{p.label}</option>
            ))}
          </select>

          <label>Stages</label>
          <div className="checkbox-group compress-stages">
            {allStages.map((stage) => (
              <label key={stage} title={stageHelp[stage]}>
                <input
                  type="checkbox"
                  checked={selectedStages.includes(stage)}
                  onChange={() => toggleStage(stage)}
                />
                {stage.replace(/_/g, " ")}
                {stageHelp[stage] && (
                  <span className="muted-text compress-stage-hint">{stageHelp[stage]}</span>
                )}
              </label>
            ))}
          </div>

          <h3 className="section-title">Models</h3>
          <label>Teacher model</label>
          <input value={teacherModel} onChange={(e) => setTeacherModel(e.target.value)} />

          <label>Student model</label>
          <input value={studentModel} onChange={(e) => setStudentModel(e.target.value)} />

          <label>Starting model dir (optional — for prune/finetune presets)</label>
          <input value={modelDir} onChange={(e) => setModelDir(e.target.value)} placeholder="~/.seiso/checkpoints/…" />

          <label>Link training job ID (optional)</label>
          <input value={linkTrainingJob} onChange={(e) => setLinkTrainingJob(e.target.value)} placeholder="uuid from Train page" />

          <h3 className="section-title">Training & pruning</h3>
          <div className="option-grid">
            <div>
              <label>Distill steps</label>
              <input
                type="number"
                min={1}
                value={distillSteps}
                onChange={(e) => setDistillSteps(e.target.value ? +e.target.value : "")}
                placeholder="preset default"
              />
            </div>
            <div>
              <label>Finetune steps</label>
              <input
                type="number"
                min={1}
                value={finetuneSteps}
                onChange={(e) => setFinetuneSteps(e.target.value ? +e.target.value : "")}
                placeholder="preset default"
              />
            </div>
          </div>

          <div className="option-grid">
            <div>
              <label>Prune ratio: {pruneRatio.toFixed(2)}</label>
              <input
                type="range"
                min={0.05}
                max={0.5}
                step={0.05}
                value={pruneRatio}
                onChange={(e) => setPruneRatio(+e.target.value)}
              />
            </div>
            <div>
              <label>Prune method</label>
              <select value={pruneMethod} onChange={(e) => setPruneMethod(e.target.value)}>
                <option value="magnitude">Magnitude</option>
                <option value="wanda">Wanda</option>
              </select>
            </div>
          </div>

          <label>Max train samples (override)</label>
          <input
            type="number"
            min={1}
            value={maxTrainSamples}
            onChange={(e) => setMaxTrainSamples(e.target.value ? +e.target.value : "")}
            placeholder="preset default"
          />

          <h3 className="section-title">Export & quantization</h3>
          <label>Export model name</label>
          <input value={exportModelName} onChange={(e) => setExportModelName(e.target.value)} />

          <label>Calibration samples (GPTQ / AWQ)</label>
          <input
            type="number"
            min={1}
            value={calibrationSamples}
            onChange={(e) => setCalibrationSamples(e.target.value ? +e.target.value : "")}
            placeholder="preset default"
          />

          <h3 className="section-title">Reproducibility</h3>
          <div className="option-grid">
            <div>
              <label>Seed</label>
              <input
                type="number"
                min={0}
                value={seed}
                onChange={(e) => setSeed(+e.target.value)}
              />
            </div>
            <div className="checkbox-group" style={{ margin: 0, justifyContent: "flex-end" }}>
              <label>
                <input
                  type="checkbox"
                  checked={deterministic}
                  onChange={(e) => setDeterministic(e.target.checked)}
                />
                Deterministic mode
              </label>
            </div>
          </div>

          <details className="config-advanced">
            <summary>Advanced options</summary>
            <label>Config file path (optional JSON override)</label>
            <input
              value={configFile}
              onChange={(e) => setConfigFile(e.target.value)}
              placeholder="~/.seiso/configs/compress.json"
            />
          </details>

          <button className="btn btn-primary btn-lg studio-action-bar-standalone" onClick={start} disabled={starting}>
            {starting ? "Starting…" : "Run compression pipeline"}
          </button>
        </div>

        <div className="card">
          <h3 className="section-title">
            Job log {activeJob ? <span className="badge">{activeJob.slice(0, 8)}</span> : ""}
          </h3>
          <div className="log-panel log-panel-tall">{logs.join("\n") || "Logs appear here during the pipeline."}</div>
          {result && (
            <div style={{ marginTop: "1rem" }}>
              <h3 className="section-title">Result</h3>
              <pre className="log-panel" style={{ fontSize: "0.8rem" }}>
                {JSON.stringify(result, null, 2)}
              </pre>
            </div>
          )}
        </div>
      </div>

      <div className="card" style={{ marginTop: "1rem" }}>
        <h3 className="section-title">Recent jobs</h3>
        {jobs.length === 0 ? (
          <p className="muted-text">No compression jobs yet.</p>
        ) : (
          <table>
            <thead>
              <tr>
                <th>ID</th>
                <th>Status</th>
                <th>Stages</th>
                <th>Model</th>
                <th>Created</th>
              </tr>
            </thead>
            <tbody>
              {jobs.map((j) => (
                <tr key={j.id}>
                  <td className="mono">{j.id.slice(0, 8)}…</td>
                  <td><span className={`badge badge-${j.status}`}>{j.status}</span></td>
                  <td>{j.stages?.join(", ") || "—"}</td>
                  <td className="mono">{j.model_dir ? j.model_dir.split("/").slice(-2).join("/") : "—"}</td>
                  <td className="muted-cell">{j.created_at?.slice(0, 19)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </StudioPageShell>
  );
}
