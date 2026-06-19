import { useEffect, useRef, useState } from "react";
import { api, RLQuantJob, RLQuantPreset, subscribeSSE } from "@/lib/api";
import { appendBoundedLog } from "@/lib/api/sse";
import { RL_QUANT_FALLBACK_PRESETS, RL_QUANT_PRESET_HINTS } from "@/lib/rlQuantPresets";
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
  theta_kernel_speedup: 0.12,
  iota_kernel_latency: 0.008,
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
  const [kernelRlEnabled, setKernelRlEnabled] = useState(false);
  const [kernelLiveBenchmark, setKernelLiveBenchmark] = useState(false);
  const [kernelHiddenDim, setKernelHiddenDim] = useState(4096);
  const [kernelBatchRows, setKernelBatchRows] = useState(4096);
  const [reward, setReward] = useState(DEFAULT_REWARD);
  const [logs, setLogs] = useState<string[]>([]);
  const [recommendation, setRecommendation] = useState<Record<string, unknown> | null>(null);
  const [activeJob, setActiveJob] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const streamAbortRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    api.listRLQuantJobs().then(setJobs).catch(console.error);
    api.rlQuantPresets().then((r) => setPresets(r.presets)).catch(console.error);
    return () => streamAbortRef.current?.();
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
        kernel_rl_enabled: kernelRlEnabled,
        kernel_live_benchmark: kernelLiveBenchmark,
        kernel_hidden_dim: kernelRlEnabled ? kernelHiddenDim : undefined,
        kernel_batch_rows: kernelRlEnabled ? kernelBatchRows : undefined,
        reward_weights: reward,
        seed: 13,
      });
      setActiveJob(res.job_id);
      streamAbortRef.current?.();
      streamAbortRef.current = subscribeSSE(`/rl-quant/jobs/${res.job_id}/stream`, (event, data) => {
        if (event === "log") setLogs((l) => appendBoundedLog(l, data));
        if (event === "error") setLogs((l) => appendBoundedLog(l, `ERROR: ${data}`));
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

  const presetList = presets.length ? presets : RL_QUANT_FALLBACK_PRESETS;

  const selectedPreset = presetList.find((p) => p.id === preset);

  return (
    <StudioPageShell
      title="RL Quantization"
      subtitle="Adaptive quantization via reinforcement learning — train a reward-guided policy, evaluate on simulator or llama.cpp, export GGUF with recommended quant levels."
      badge={<span className="trust-badge trust-badge-dim">REINFORCE · multiseed sweeps</span>}
    >
      <div className="train-layout">
        <div className="card studio-card">
          <div className="studio-card-head">
            <span className="studio-card-icon" aria-hidden>①</span>
            <div className="studio-card-head-text">
              <div className="studio-card-title">Experiment config</div>
              <div className="studio-card-desc">Preset, backends, inputs, and reward weights</div>
            </div>
          </div>

          <FormSection title="Experiment preset" hint="Reproducible configs with logged artifacts.">
            <div className="form-field">
              <label>Preset</label>
              <select value={preset} onChange={(e) => setPreset(e.target.value)}>
                {presetList.map((p) => (
                  <option key={p.id} value={p.id}>{p.label}</option>
                ))}
              </select>
            </div>
            {RL_QUANT_PRESET_HINTS[preset] && <p className="field-hint">{RL_QUANT_PRESET_HINTS[preset]}</p>}
            {selectedPreset && (
              <p className="field-hint">
                Backend {selectedPreset.backend} · trainer {selectedPreset.training_backend}
              </p>
            )}
          </FormSection>

          <FormSection title="Training budget" collapsible defaultOpen>
            <div className="option-grid">
              <div className="form-field">
                <label>Training episodes</label>
                <input type="number" min={8} value={trainingEpisodes} onChange={(e) => setTrainingEpisodes(+e.target.value)} />
              </div>
              <div className="form-field">
                <label>Evaluation episodes</label>
                <input type="number" min={4} value={evaluationEpisodes} onChange={(e) => setEvaluationEpisodes(+e.target.value)} />
              </div>
            </div>
          </FormSection>

          <FormSection title="Backends" collapsible>
            <div className="form-field">
              <label>Measure backend</label>
              <select value={backend} onChange={(e) => setBackend(e.target.value)}>
                <option value="simulator">Simulator (no GPU)</option>
                <option value="llama_cpp">llama.cpp (GGUF path required)</option>
              </select>
            </div>
            <div className="form-field">
              <label>Policy trainer</label>
              <select value={trainingBackend} onChange={(e) => setTrainingBackend(e.target.value)}>
                <option value="stdlib">Stdlib REINFORCE</option>
                <option value="pytorch">PyTorch / CUDA</option>
              </select>
            </div>
          </FormSection>

          <FormSection title="Inputs" collapsible defaultOpen={false}>
            <div className="form-field">
              <label>Fine-tune checkpoint</label>
              <input value={checkpoint} onChange={(e) => setCheckpoint(e.target.value)} placeholder="~/.seiso/checkpoints/…" />
            </div>
            <div className="form-field">
              <label>GGUF path (llama.cpp)</label>
              <input value={ggufPath} onChange={(e) => setGgufPath(e.target.value)} placeholder="models/model-q4.gguf" />
            </div>
            <div className="form-field">
              <label>Link training job ID</label>
              <input value={linkTrainingJob} onChange={(e) => setLinkTrainingJob(e.target.value)} placeholder="uuid from Train page" />
            </div>
          </FormSection>

          <FormSection title="CUDA kernel RL" hint="Co-train quantization with fused CUDA launch profiles." collapsible defaultOpen={false}>
            <div className="studio-checkbox-grid">
              <label className="studio-checkbox-item">
                <input
                  type="checkbox"
                  checked={kernelRlEnabled}
                  onChange={(e) => setKernelRlEnabled(e.target.checked)}
                />
                Enable kernel RL (joint quant + CUDA profile policy)
              </label>
              <label className="studio-checkbox-item">
                <input
                  type="checkbox"
                  checked={kernelLiveBenchmark}
                  onChange={(e) => setKernelLiveBenchmark(e.target.checked)}
                  disabled={!kernelRlEnabled}
                />
                Live CUDA micro-benchmarks (NVIDIA GPU; slower, ground-truth)
              </label>
            </div>
            {kernelRlEnabled && (
              <div className="option-grid">
                <div className="form-field">
                  <label>Hidden dim (bench)</label>
                  <input
                    type="number"
                    min={128}
                    step={128}
                    value={kernelHiddenDim}
                    onChange={(e) => setKernelHiddenDim(+e.target.value)}
                  />
                </div>
                <div className="form-field">
                  <label>Batch rows (bench)</label>
                  <input
                    type="number"
                    min={64}
                    step={64}
                    value={kernelBatchRows}
                    onChange={(e) => setKernelBatchRows(+e.target.value)}
                  />
                </div>
              </div>
            )}
            <p className="field-hint">
              Profiles: auto, stripe, parallax, narrow_opt, wide_throughput, balanced. Winning profile is applied to native CUDA kernels at runtime.
            </p>
          </FormSection>

          <FormSection title="Export options" hint="Optional GGUF export and MoE variants." collapsible defaultOpen>
            <div className="studio-checkbox-grid">
              <label className="studio-checkbox-item">
                <input type="checkbox" checked={ggufExport} onChange={(e) => setGgufExport(e.target.checked)} />
                Export GGUF after recommendation
              </label>
              <label className="studio-checkbox-item">
                <input type="checkbox" checked={moeEnabled} onChange={(e) => setMoeEnabled(e.target.checked)} />
                MoE expert variants
              </label>
            </div>
          </FormSection>

          <FormSection title="Reward engineering" hint="Tune the multi-objective reward surface." collapsible>
            <RewardWeights weights={reward} onChange={(w) => setReward({ ...reward, ...w })} />
          </FormSection>
        </div>

        <div className="card studio-card">
          <div className="studio-card-head">
            <span className="studio-card-icon" aria-hidden>②</span>
            <div className="studio-card-head-text">
              <div className="studio-card-title">Live output</div>
              <div className="studio-card-desc">Streaming logs and recommendation artifacts</div>
              {activeJob && (
                <div className="studio-card-meta">
                  <span className="mono studio-job-id">{activeJob.slice(0, 8)}…</span>
                </div>
              )}
            </div>
          </div>
          <LogStream
            logs={logs}
            emptyMessage="Run the RL quant pipeline to stream logs here."
            tall
          />
          {recommendation && (
            <div className="studio-artifact-section">
              <div className="form-section-head">
                <h3 className="form-section-title">Recommendation artifact</h3>
                <p className="form-section-hint">Policy output from the completed sweep</p>
              </div>
              {(() => {
                const decision = (recommendation.decision ?? recommendation.recommended_quant) as
                  | Record<string, unknown>
                  | undefined;
                const kernelName =
                  (decision?.kernel_profile_name as string | undefined) ||
                  ((decision?.metadata as Record<string, unknown> | undefined)?.kernel_profile_name as
                    | string
                    | undefined);
                return kernelName ? (
                  <p className="field-hint">
                    Recommended CUDA kernel profile: <span className="mono">{kernelName}</span>
                  </p>
                ) : null;
              })()}
              <div className="studio-artifact-panel">
                <ArtifactViewer data={recommendation} />
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="card studio-card studio-run-card">
        <div className="studio-card-head studio-card-head-inline">
          <span className="studio-card-icon" aria-hidden>▶</span>
          <div className="studio-card-head-text">
            <div className="studio-card-title">Run pipeline</div>
            <div className="studio-card-desc">
              Starts a local REINFORCE sweep and streams metrics to the log panel.
            </div>
          </div>
        </div>
        <div className="studio-action-bar studio-action-bar-flush">
          <button className="btn btn-primary btn-lg" onClick={start} disabled={starting}>
            {starting ? "Starting…" : "Run RL quant pipeline"}
          </button>
        </div>
      </div>

      <div className="card studio-card">
        <div className="studio-card-head">
          <span className="studio-card-icon" aria-hidden>③</span>
          <div className="studio-card-head-text">
            <div className="studio-card-title">Recent experiments</div>
            <div className="studio-card-desc">Past RL quant jobs on this machine</div>
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
              header: "ID",
              mono: true,
              render: (j) => `${j.id.slice(0, 8)}…`,
            },
            {
              key: "status",
              header: "Status",
              render: (j) => <span className={`badge badge-${j.status}`}>{j.status}</span>,
            },
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
          emptyMessage="No RL quant jobs yet — configure settings above and run the pipeline."
        />
        <p className="field-hint studio-field-hint">
          Completed job IDs can be used on the Export page for RL-recommended GGUF quantizations.
        </p>
      </div>
    </StudioPageShell>
  );
}
