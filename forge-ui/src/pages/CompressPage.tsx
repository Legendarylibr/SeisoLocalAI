import { useEffect, useState } from "react";
import { api, CompressJob, CompressPreset, subscribeSSE } from "@/lib/api";

export function CompressPage() {
  const [jobs, setJobs] = useState<CompressJob[]>([]);
  const [presets, setPresets] = useState<CompressPreset[]>([]);
  const [preset, setPreset] = useState("smoke");
  const [teacherModel, setTeacherModel] = useState("codellama/CodeLlama-13b-hf");
  const [studentModel, setStudentModel] = useState("codellama/CodeLlama-7b-hf");
  const [modelDir, setModelDir] = useState("");
  const [distillSteps, setDistillSteps] = useState<number | "">("");
  const [finetuneSteps, setFinetuneSteps] = useState<number | "">("");
  const [pruneRatio, setPruneRatio] = useState(0.25);
  const [linkTrainingJob, setLinkTrainingJob] = useState("");
  const [logs, setLogs] = useState<string[]>([]);
  const [result, setResult] = useState<Record<string, unknown> | null>(null);
  const [activeJob, setActiveJob] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);

  useEffect(() => {
    api.listCompressJobs().then(setJobs).catch(console.error);
    api.compressPresets().then((r) => setPresets(r.presets)).catch(console.error);
  }, []);

  const refreshJobs = () => api.listCompressJobs().then(setJobs).catch(console.error);

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
        seed: 42,
      };
      if (modelDir) body.model_dir = modelDir;
      if (distillSteps !== "") body.distill_steps = distillSteps;
      if (finetuneSteps !== "") body.finetune_steps = finetuneSteps;
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

  const selectedPreset = presets.find((p) => p.id === preset);

  return (
    <div>
      <h1 className="page-title">Model Compression</h1>
      <p className="page-sub">
        Code Llama compression pipeline: distillation, MLP pruning, recovery fine-tune, evaluation,
        and export bundles (vLLM/Docker/GGUF scripts). Hash-chained manifests for reproducibility.
      </p>

      <div className="train-layout">
        <div className="card">
          <label>Preset</label>
          <select value={preset} onChange={(e) => setPreset(e.target.value)}>
            {(presets.length ? presets : [
              { id: "smoke", label: "Smoke", stages: ["distill", "prune", "finetune", "evaluate", "export"] },
              { id: "full", label: "Full", stages: [] },
              { id: "distill_only", label: "Distill Only", stages: [] },
              { id: "prune_recover", label: "Prune Recover", stages: [] },
            ]).map((p) => (
              <option key={p.id} value={p.id}>{p.label}</option>
            ))}
          </select>

          {selectedPreset && selectedPreset.stages.length > 0 && (
            <p className="page-sub" style={{ marginTop: "0.5rem" }}>
              Stages: {selectedPreset.stages.join(" → ")}
            </p>
          )}

          <label>Teacher model</label>
          <input value={teacherModel} onChange={(e) => setTeacherModel(e.target.value)} />

          <label>Student model</label>
          <input value={studentModel} onChange={(e) => setStudentModel(e.target.value)} />

          <label>Starting model dir (optional — for prune/finetune presets)</label>
          <input value={modelDir} onChange={(e) => setModelDir(e.target.value)} placeholder="~/.seiso/checkpoints/…" />

          <label>Distill steps (override)</label>
          <input
            type="number"
            min={1}
            value={distillSteps}
            onChange={(e) => setDistillSteps(e.target.value ? +e.target.value : "")}
            placeholder="preset default"
          />

          <label>Finetune steps (override)</label>
          <input
            type="number"
            min={1}
            value={finetuneSteps}
            onChange={(e) => setFinetuneSteps(e.target.value ? +e.target.value : "")}
            placeholder="preset default"
          />

          <label>Prune ratio</label>
          <input
            type="number"
            min={0.05}
            max={0.5}
            step={0.05}
            value={pruneRatio}
            onChange={(e) => setPruneRatio(+e.target.value)}
          />

          <label>Link training job ID (optional)</label>
          <input value={linkTrainingJob} onChange={(e) => setLinkTrainingJob(e.target.value)} placeholder="uuid from Train page" />

          <button className="btn btn-primary" style={{ marginTop: "1rem" }} onClick={start} disabled={starting}>
            {starting ? "Starting…" : "Run compression pipeline"}
          </button>
        </div>

        <div className="card">
          <h3>Job log {activeJob ? `(${activeJob.slice(0, 8)}…)` : ""}</h3>
          <div className="log-panel">{logs.join("\n") || "Logs appear here during the pipeline."}</div>
          {result && (
            <div style={{ marginTop: "1rem" }}>
              <h3>Result</h3>
              <pre className="log-panel" style={{ fontSize: "0.8rem" }}>
                {JSON.stringify(result, null, 2)}
              </pre>
            </div>
          )}
        </div>
      </div>

      <div className="card" style={{ marginTop: "1rem" }}>
        <h3>Recent jobs</h3>
        <table style={{ width: "100%", fontSize: "0.9rem" }}>
          <thead>
            <tr>
              <th align="left">ID</th>
              <th align="left">Status</th>
              <th align="left">Stages</th>
              <th align="left">Model</th>
              <th align="left">Created</th>
            </tr>
          </thead>
          <tbody>
            {jobs.map((j) => (
              <tr key={j.id}>
                <td><code>{j.id.slice(0, 8)}</code></td>
                <td>{j.status}</td>
                <td>{j.stages?.join(", ") || "—"}</td>
                <td><code>{j.model_dir ? j.model_dir.split("/").slice(-2).join("/") : "—"}</code></td>
                <td>{j.created_at?.slice(0, 19)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
