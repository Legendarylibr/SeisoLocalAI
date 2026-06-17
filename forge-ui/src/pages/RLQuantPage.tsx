import { useEffect, useState } from "react";
import { api, RLQuantJob, RLQuantPreset, subscribeSSE } from "@/lib/api";
import { StudioPageShell } from "@/components/StudioPageShell";

const DEFAULT_REWARD = {
  alpha_latency: 0.02,
  beta_throughput: 0.06,
  gamma_perplexity: 0.85,
  delta_memory: 0.002,
  epsilon_instability: 1.0,
};

export function RLQuantPage() {
  const [jobs, setJobs] = useState<RLQuantJob[]>([]);
  const [presets, setPresets] = useState<RLQuantPreset[]>([]);
  const [preset, setPreset] = useState("minimal");
  const [trainingEpisodes, setTrainingEpisodes] = useState(256);
  const [evaluationEpisodes, setEvaluationEpisodes] = useState(64);
  const [backend, setBackend] = useState("simulator");
  const [trainingBackend, setTrainingBackend] = useState("stdlib");
  const [checkpoint, setCheckpoint] = useState("");
  const [ggufPath, setGgufPath] = useState("");
  const [linkTrainingJob, setLinkTrainingJob] = useState("");
  const [ggufExport, setGgufExport] = useState(false);
  const [moeEnabled, setMoeEnabled] = useState(false);
  const [reward, setReward] = useState(DEFAULT_REWARD);
  const [logs, setLogs] = useState<string[]>([]);
  const [recommendation, setRecommendation] = useState<Record<string, unknown> | null>(null);
  const [activeJob, setActiveJob] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);

  useEffect(() => {
    api.listRLQuantJobs().then(setJobs).catch(console.error);
    api.rlQuantPresets().then((r) => setPresets(r.presets)).catch(console.error);
  }, []);

  const refreshJobs = () => api.listRLQuantJobs().then(setJobs).catch(console.error);

  const start = async () => {
    setStarting(true);
    setLogs([]);
    setRecommendation(null);
    try {
      const res = await api.startRLQuant({
        preset,
        training_episodes: trainingEpisodes,
        evaluation_episodes: evaluationEpisodes,
        backend,
        training_backend: trainingBackend,
        checkpoint_path: checkpoint || undefined,
        gguf_path: ggufPath || undefined,
        link_training_job_id: linkTrainingJob || undefined,
        gguf_export: ggufExport,
        moe_enabled: moeEnabled,
        reward_weights: reward,
        seed: 13,
      });
      setActiveJob(res.job_id);
      subscribeSSE(`/rl-quant/jobs/${res.job_id}/stream`, (event, data) => {
        if (event === "log") setLogs((l) => [...l, data]);
        if (event === "error") setLogs((l) => [...l, `ERROR: ${data}`]);
        if (event === "recommendation") {
          try {
            setRecommendation(JSON.parse(data));
          } catch {
            /* ignore */
          }
        }
        if (event === "result") refreshJobs();
      });
      refreshJobs();
    } finally {
      setStarting(false);
    }
  };

  const selectedPreset = presets.find((p) => p.id === preset);

  return (
    <StudioPageShell
      title="RL Quantization"
      subtitle="Adaptive quantization via reinforcement learning and reward engineering (llama.cpp / simulator). Train a policy, get a GGUF quant recommendation, then export or chat."
    >
      <div className="train-layout">
        <div className="card">
          <label>Preset</label>
          <select value={preset} onChange={(e) => setPreset(e.target.value)}>
            {(presets.length ? presets : [{ id: "minimal", label: "Minimal" }, { id: "reproducible", label: "Reproducible" }, { id: "post_train", label: "Post-train" }]).map((p) => (
              <option key={p.id} value={p.id}>{p.label}</option>
            ))}
          </select>

          <label>Training episodes</label>
          <input type="number" min={8} value={trainingEpisodes} onChange={(e) => setTrainingEpisodes(+e.target.value)} />

          <label>Evaluation episodes</label>
          <input type="number" min={4} value={evaluationEpisodes} onChange={(e) => setEvaluationEpisodes(+e.target.value)} />

          <label>Measure backend</label>
          <select value={backend} onChange={(e) => setBackend(e.target.value)}>
            <option value="simulator">Simulator (default, no GPU)</option>
            <option value="llama_cpp">llama.cpp (GGUF path required)</option>
          </select>

          <label>Policy trainer</label>
          <select value={trainingBackend} onChange={(e) => setTrainingBackend(e.target.value)}>
            <option value="stdlib">Stdlib REINFORCE (research default)</option>
            <option value="pytorch">PyTorch / CUDA (optional)</option>
          </select>

          <label>Fine-tune checkpoint (optional)</label>
          <input value={checkpoint} onChange={(e) => setCheckpoint(e.target.value)} placeholder="~/.seiso/checkpoints/…" />

          <label>GGUF path for llama.cpp backend (optional)</label>
          <input value={ggufPath} onChange={(e) => setGgufPath(e.target.value)} placeholder="models/model-q4.gguf" />

          <label>Link training job ID (optional)</label>
          <input value={linkTrainingJob} onChange={(e) => setLinkTrainingJob(e.target.value)} placeholder="uuid from Train page" />

          <label style={{ display: "flex", alignItems: "center", gap: "0.4rem" }}>
            <input type="checkbox" checked={ggufExport} onChange={(e) => setGgufExport(e.target.checked)} />
            Export GGUF after recommendation (llama.cpp quantize)
          </label>
          <label style={{ display: "flex", alignItems: "center", gap: "0.4rem" }}>
            <input type="checkbox" checked={moeEnabled} onChange={(e) => setMoeEnabled(e.target.checked)} />
            MoE expert variants
          </label>

          <h3 style={{ marginTop: "1rem" }}>Reward weights</h3>
          {Object.entries(reward).map(([key, val]) => (
            <div key={key}>
              <label>{key}</label>
              <input
                type="number"
                step="0.001"
                value={val}
                onChange={(e) => setReward((r) => ({ ...r, [key]: +e.target.value }))}
              />
            </div>
          ))}

          {selectedPreset && (
            <p className="page-sub" style={{ marginTop: "0.75rem" }}>
              {selectedPreset.label}: backend {selectedPreset.backend}, trainer {selectedPreset.training_backend}
            </p>
          )}

          <button className="btn btn-primary" style={{ marginTop: "1rem" }} onClick={start} disabled={starting}>
            {starting ? "Starting…" : "Run RL quant pipeline"}
          </button>
        </div>

        <div className="card">
          <h3>Job log {activeJob ? `(${activeJob.slice(0, 8)}…)` : ""}</h3>
          <div className="log-panel">{logs.join("\n") || "Logs appear here during training."}</div>
          {recommendation && (
            <div style={{ marginTop: "1rem" }}>
              <h3>Recommendation</h3>
              <pre className="log-panel" style={{ fontSize: "0.8rem" }}>
                {JSON.stringify(recommendation, null, 2)}
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
              <th align="left">GGUF quants</th>
              <th align="left">Created</th>
            </tr>
          </thead>
          <tbody>
            {jobs.map((j) => (
              <tr key={j.id}>
                <td><code>{j.id.slice(0, 8)}</code></td>
                <td>{j.status}</td>
                <td>{j.gguf_quants?.join(", ") || "—"}</td>
                <td>{j.created_at?.slice(0, 19)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="page-sub" style={{ marginTop: "0.75rem" }}>
          Use a completed job ID on the Export page to apply RL-recommended GGUF quantizations.
        </p>
      </div>
    </StudioPageShell>
  );
}
