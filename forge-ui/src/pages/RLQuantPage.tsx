import { useEffect, useState } from "react";
import { api, RLQuantJob, RLQuantPreset, subscribeSSE } from "@/lib/api";
import { StudioPageShell } from "@/components/StudioPageShell";
import { FormSection } from "@/components/research/FormSection";
import { RewardWeights } from "@/components/research/RewardWeights";
import { ArtifactViewer } from "@/components/research/ArtifactViewer";
import { DataTable } from "@/components/research/DataTable";
import { LogStream } from "@/components/research/LogStream";

const DEFAULT_REWARD = {
  alpha_latency: 0.02,
  beta_throughput: 0.06,
  gamma_perplexity: 0.85,
  delta_memory: 0.002,
  epsilon_instability: 1.0,
};

const PRESET_HINTS: Record<string, string> = {
  minimal: "Fast smoke run — simulator backend, few episodes.",
  reproducible: "Fixed seeds and logged artifacts for paper-grade reproducibility.",
  post_train: "Post fine-tune checkpoint — links training output to quant recommendation.",
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

  const presetList = presets.length
    ? presets
    : [
        { id: "minimal", label: "Minimal", backend: "simulator", training_backend: "stdlib" },
        { id: "reproducible", label: "Reproducible", backend: "simulator", training_backend: "stdlib" },
        { id: "post_train", label: "Post-train", backend: "simulator", training_backend: "stdlib" },
      ];

  const selectedPreset = presetList.find((p) => p.id === preset);

  return (
    <StudioPageShell
      title="RL Quantization"
      subtitle="Adaptive quantization via reinforcement learning — train a reward-guided policy, evaluate on simulator or llama.cpp, export GGUF with recommended quant levels."
      badge={<span className="trust-badge trust-badge-dim">REINFORCE · multiseed sweeps</span>}
    >
      <div className="train-layout">
        <div className="card research-config-card">
          <FormSection title="Experiment preset" hint="Reproducible configs with logged artifacts.">
            <label>Preset</label>
            <select value={preset} onChange={(e) => setPreset(e.target.value)}>
              {presetList.map((p) => (
                <option key={p.id} value={p.id}>{p.label}</option>
              ))}
            </select>
            {PRESET_HINTS[preset] && <p className="field-hint">{PRESET_HINTS[preset]}</p>}
            {selectedPreset && (
              <p className="field-hint">
                Backend {selectedPreset.backend} · trainer {selectedPreset.training_backend}
              </p>
            )}
          </FormSection>

          <FormSection title="Training budget" collapsible defaultOpen>
            <div className="option-grid">
              <div>
                <label>Training episodes</label>
                <input type="number" min={8} value={trainingEpisodes} onChange={(e) => setTrainingEpisodes(+e.target.value)} />
              </div>
              <div>
                <label>Evaluation episodes</label>
                <input type="number" min={4} value={evaluationEpisodes} onChange={(e) => setEvaluationEpisodes(+e.target.value)} />
              </div>
            </div>
          </FormSection>

          <FormSection title="Backends" collapsible>
            <label>Measure backend</label>
            <select value={backend} onChange={(e) => setBackend(e.target.value)}>
              <option value="simulator">Simulator (no GPU)</option>
              <option value="llama_cpp">llama.cpp (GGUF path required)</option>
            </select>

            <label>Policy trainer</label>
            <select value={trainingBackend} onChange={(e) => setTrainingBackend(e.target.value)}>
              <option value="stdlib">Stdlib REINFORCE</option>
              <option value="pytorch">PyTorch / CUDA</option>
            </select>
          </FormSection>

          <FormSection title="Inputs" collapsible defaultOpen={false}>
            <label>Fine-tune checkpoint</label>
            <input value={checkpoint} onChange={(e) => setCheckpoint(e.target.value)} placeholder="~/.seiso/checkpoints/…" />

            <label>GGUF path (llama.cpp)</label>
            <input value={ggufPath} onChange={(e) => setGgufPath(e.target.value)} placeholder="models/model-q4.gguf" />

            <label>Link training job ID</label>
            <input value={linkTrainingJob} onChange={(e) => setLinkTrainingJob(e.target.value)} placeholder="uuid from Train page" />
          </FormSection>

          <div className="checkbox-group">
            <label>
              <input type="checkbox" checked={ggufExport} onChange={(e) => setGgufExport(e.target.checked)} />
              Export GGUF after recommendation
            </label>
            <label>
              <input type="checkbox" checked={moeEnabled} onChange={(e) => setMoeEnabled(e.target.checked)} />
              MoE expert variants
            </label>
          </div>

          <FormSection title="Reward engineering" hint="Tune the multi-objective reward surface." collapsible>
            <RewardWeights weights={reward} onChange={(w) => setReward({ ...reward, ...w })} />
          </FormSection>

          <button className="btn btn-primary btn-lg studio-action-bar-standalone" onClick={start} disabled={starting}>
            {starting ? "Starting…" : "Run RL quant pipeline"}
          </button>
        </div>

        <div className="card research-config-card">
          <LogStream
            title={activeJob ? `Job log (${activeJob.slice(0, 8)}…)` : "Job log"}
            logs={logs}
            tall
          />
          {recommendation && (
            <div className="artifact-section">
              <h3 className="section-title">Recommendation artifact</h3>
              <ArtifactViewer data={recommendation} />
            </div>
          )}
        </div>
      </div>

      <div className="card">
        <h3 className="section-title">Recent experiments</h3>
        <DataTable
          columns={[
            {
              key: "id",
              header: "ID",
              mono: true,
              render: (j) => j.id.slice(0, 8),
            },
            { key: "status", header: "Status" },
            {
              key: "gguf_quants",
              header: "GGUF quants",
              render: (j) => j.gguf_quants?.join(", ") || "—",
            },
            {
              key: "created_at",
              header: "Created",
              render: (j) => j.created_at?.slice(0, 19) ?? "—",
            },
          ]}
          rows={jobs}
          getRowKey={(j) => j.id}
          emptyMessage="No RL quant jobs yet."
        />
        <p className="field-hint" style={{ marginTop: "0.75rem" }}>
          Completed job IDs can be used on the Export page for RL-recommended GGUF quantizations.
        </p>
      </div>
    </StudioPageShell>
  );
}
