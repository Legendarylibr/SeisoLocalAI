import { useEffect, useState } from "react";
import { api, ImageCompressJob, ImageCompressPreset, subscribeSSE } from "@/lib/api";
import { StudioPageShell } from "@/components/StudioPageShell";

export function ImageCompressPage() {
  const [jobs, setJobs] = useState<ImageCompressJob[]>([]);
  const [presets, setPresets] = useState<ImageCompressPreset[]>([]);
  const [preset, setPreset] = useState("smoke");
  const [baseModel, setBaseModel] = useState("runwayml/stable-diffusion-v1-5");
  const [modelDir, setModelDir] = useState("");
  const [dataPath, setDataPath] = useState("");
  const [steps, setSteps] = useState<number | "">("");
  const [finetuneSteps, setFinetuneSteps] = useState<number | "">("");
  const [pruneRatio, setPruneRatio] = useState(0.3);
  const [logs, setLogs] = useState<string[]>([]);
  const [result, setResult] = useState<Record<string, unknown> | null>(null);
  const [activeJob, setActiveJob] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);

  useEffect(() => {
    api.listImageCompressJobs().then(setJobs).catch(console.error);
    api.imageCompressPresets().then((r) => setPresets(r.presets)).catch(console.error);
  }, []);

  const refreshJobs = () => api.listImageCompressJobs().then(setJobs).catch(console.error);

  const start = async () => {
    setStarting(true);
    setLogs([]);
    setResult(null);
    try {
      const body: Record<string, unknown> = {
        preset,
        base_model: baseModel,
        prune_ratio: pruneRatio,
      };
      if (modelDir) body.model_dir = modelDir;
      if (dataPath) body.data_path = dataPath;
      if (steps !== "") body.steps = steps;
      if (finetuneSteps !== "") body.finetune_steps = finetuneSteps;

      const res = await api.startImageCompress(body);
      setActiveJob(res.job_id);
      subscribeSSE(`/image-compress/jobs/${res.job_id}/stream`, (event, data) => {
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
    <StudioPageShell
      title="Image Model Compression"
      subtitle="Stable Diffusion compression pipeline: progressive distillation, structured pruning, recovery fine-tuning, INT8 quantisation, and export. Quality tracked via CLIP/LPIPS/SSIM."
    >
      <div className="train-layout">
        <div className="card">
          <label>Preset</label>
          <select value={preset} onChange={(e) => setPreset(e.target.value)}>
            {(presets.length ? presets : [
              { id: "smoke", label: "Smoke", stages: ["baseline", "distill_progressive", "prune", "quantize", "evaluate_quantized", "report"] },
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

          <label>Base model</label>
          <input value={baseModel} onChange={(e) => setBaseModel(e.target.value)} />

          <label>Starting model dir (optional — for prune/finetune presets)</label>
          <input value={modelDir} onChange={(e) => setModelDir(e.target.value)} placeholder="~/.seiso/image_compress/…" />

          <label>Captions JSON path (optional — auto-generated if empty)</label>
          <input value={dataPath} onChange={(e) => setDataPath(e.target.value)} placeholder="~/.seiso/…/captions.json" />

          <label>Distill steps (override)</label>
          <input
            type="number"
            min={1}
            value={steps}
            onChange={(e) => setSteps(e.target.value ? +e.target.value : "")}
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

          <button className="btn btn-primary" style={{ marginTop: "1rem" }} onClick={start} disabled={starting}>
            {starting ? "Starting…" : "Run image compression pipeline"}
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
    </StudioPageShell>
  );
}
