import { useEffect, useState } from "react";
import { api, RLQuantJob, subscribeSSE } from "@/lib/api";

export function ExportPage() {
  const [checkpoint, setCheckpoint] = useState("");
  const [formats, setFormats] = useState(["merged", "gguf"]);
  const [hubRepo, setHubRepo] = useState("");
  const [rlQuantJobId, setRlQuantJobId] = useState("");
  const [rlJobs, setRlJobs] = useState<RLQuantJob[]>([]);
  const [logs, setLogs] = useState<string[]>([]);

  useEffect(() => {
    api.listRLQuantJobs().then(setRlJobs).catch(console.error);
  }, []);

  const completedRlJobs = rlJobs.filter((j) => j.status === "completed" && j.gguf_quants?.length);

  const start = async () => {
    if (!checkpoint.trim()) return;
    setLogs([]);
    const res = await api.startExport(
      checkpoint,
      formats,
      hubRepo || undefined,
      rlQuantJobId || undefined,
    );
    subscribeSSE(`/export/jobs/${res.job_id}/stream`, (event, data) => {
      if (event === "log" || event === "result") setLogs((l) => [...l, data]);
    });
  };

  const toggleFormat = (f: string) => {
    setFormats((prev) => (prev.includes(f) ? prev.filter((x) => x !== f) : [...prev, f]));
  };

  return (
    <div>
      <h1 className="page-title">Export</h1>
      <p className="page-sub">
        Merge LoRA, quantize GGUF, generate Ollama Modelfile — or apply RL-recommended quants from RL Quant jobs.
      </p>

      <div className="card">
        <label>Checkpoint path</label>
        <input value={checkpoint} onChange={(e) => setCheckpoint(e.target.value)} placeholder="~/.seiso/checkpoints/…" />
        <label>Formats</label>
        <div style={{ display: "flex", gap: "0.5rem", marginBottom: "0.75rem" }}>
          {["merged", "lora", "gguf"].map((f) => (
            <button
              key={f}
              type="button"
              className={`btn ${formats.includes(f) ? "btn-primary" : ""}`}
              onClick={() => toggleFormat(f)}
            >
              {f}
            </button>
          ))}
        </div>
        <label>RL quant job (optional — overrides GGUF quants)</label>
        <select value={rlQuantJobId} onChange={(e) => setRlQuantJobId(e.target.value)}>
          <option value="">Manual / default quants</option>
          {completedRlJobs.map((j) => (
            <option key={j.id} value={j.id}>
              {j.id.slice(0, 8)} — {j.gguf_quants.join(", ")}
            </option>
          ))}
        </select>
        <label>Hub repo (optional)</label>
        <input value={hubRepo} onChange={(e) => setHubRepo(e.target.value)} placeholder="username/model-name" />
        <button className="btn btn-primary" onClick={start}>Start export</button>
      </div>

      {logs.length > 0 && (
        <div className="card">
          <h3>Export log</h3>
          <div className="log-panel">{logs.join("\n")}</div>
        </div>
      )}
    </div>
  );
}
