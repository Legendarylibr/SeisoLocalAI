import { useEffect, useState } from "react";
import { api, ExportJob, HubPublishFields, PublishableModel, RLQuantJob, subscribeSSE } from "@/lib/api";

function emptyHub(): HubPublishFields {
  return { username: "", model_name: "", author: "", license: "apache-2.0", description: "", use_cli: false };
}

async function saveBlobResponse(res: Response, fallbackName: string) {
  if (!res.ok) throw new Error("Download failed");
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  const cd = res.headers.get("content-disposition");
  const match = cd?.match(/filename="?([^";]+)"?/);
  a.download = match?.[1] || fallbackName;
  a.click();
  URL.revokeObjectURL(url);
}

export function ExportPage() {
  const [checkpoint, setCheckpoint] = useState("");
  const [formats, setFormats] = useState(["merged", "gguf"]);
  const [hub, setHub] = useState<HubPublishFields>(emptyHub);
  const [pushOnExport, setPushOnExport] = useState(false);
  const [rlQuantJobId, setRlQuantJobId] = useState("");
  const [rlJobs, setRlJobs] = useState<RLQuantJob[]>([]);
  const [publishable, setPublishable] = useState<PublishableModel[]>([]);
  const [exportJobs, setExportJobs] = useState<ExportJob[]>([]);
  const [selectedModelId, setSelectedModelId] = useState("");
  const [selectedExportJobId, setSelectedExportJobId] = useState("");
  const [hfTokenInput, setHfTokenInput] = useState("");
  const [logs, setLogs] = useState<string[]>([]);
  const [lastExportJobId, setLastExportJobId] = useState("");
  const [profile, setProfile] = useState("");
  const [profiles, setProfiles] = useState<{ id: string; formats: string[] }[]>([]);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    api.listRLQuantJobs().then(setRlJobs).catch(console.error);
    api.listPublishableOutputs().then(setPublishable).catch(console.error);
    api.listExportJobs().then(setExportJobs).catch(console.error);
    api.listExportProfiles().then(setProfiles).catch(console.error);
  }, []);

  const completedRlJobs = rlJobs.filter((j) => j.status === "completed" && j.gguf_quants?.length);
  const completedExports = exportJobs.filter((j) => j.status === "completed");

  const hubFields = (): HubPublishFields | undefined => {
    if (!pushOnExport && !selectedModelId && !selectedExportJobId) return undefined;
    const fields: HubPublishFields = {
      ...hub,
      hf_token: hfTokenInput.trim() || undefined,
    };
    if (!fields.username || !fields.model_name || !fields.author) return undefined;
    return fields;
  };

  const start = async () => {
    if (!checkpoint.trim()) return;
    setBusy(true);
    setLogs([]);
    try {
      const res = await api.startExport(
        checkpoint,
        formats,
        pushOnExport ? hubFields() : undefined,
        rlQuantJobId || undefined,
        profile || undefined,
      );
      setLastExportJobId(res.job_id);
      subscribeSSE(`/export/jobs/${res.job_id}/stream`, (event, data) => {
        if (event === "log" || event === "result") setLogs((l) => [...l, data]);
      });
      setTimeout(() => api.listPublishableOutputs().then(setPublishable).catch(console.error), 2000);
      setTimeout(() => api.listExportJobs().then(setExportJobs).catch(console.error), 2000);
    } catch (err) {
      setLogs([(err as Error).message]);
    } finally {
      setBusy(false);
    }
  };

  const precheck = async () => {
    const fields = hubFields();
    if (!fields) {
      setLogs(["Username, model name, and author are required for Hub precheck."]);
      return;
    }
    setBusy(true);
    setLogs([]);
    try {
      const res = await api.precheckHubExport({
        hub: fields,
        formats,
        profile: profile || undefined,
      });
      const lines = [
        res.ok ? `Precheck passed for ${res.repo_id}` : `Precheck failed for ${res.repo_id}`,
        ...res.errors.map((e) => `Error: ${e}`),
        ...res.warnings.map((w) => `Warning: ${w}`),
      ];
      if (res.model_card_preview) {
        lines.push("", "--- Model card preview ---", res.model_card_preview.slice(0, 1200));
      }
      setLogs(lines);
    } catch (err) {
      setLogs([(err as Error).message]);
    } finally {
      setBusy(false);
    }
  };

  const publish = async () => {
    const fields = hubFields();
    if (!fields) {
      setLogs(["Username, model name, and author are required for Hugging Face publish."]);
      return;
    }
    if (!selectedModelId && !selectedExportJobId) {
      setLogs(["Select a Seiso export output to publish."]);
      return;
    }
    setBusy(true);
    setLogs([]);
    try {
      const res = await api.publishToHub({
        model_id: selectedModelId || undefined,
        export_job_id: selectedExportJobId || undefined,
        hub: fields,
      });
      setLogs([res.log || `Published to ${res.repo_id}`]);
    } catch (err) {
      setLogs([(err as Error).message]);
    } finally {
      setBusy(false);
    }
  };

  const downloadExport = async (jobId: string) => {
    try {
      const res = await api.downloadExportOutput(jobId);
      await saveBlobResponse(res, "model.gguf");
    } catch (err) {
      setLogs([(err as Error).message]);
    }
  };

  const downloadModel = async (modelId: string, name: string) => {
    try {
      const res = await api.downloadLocalModel(modelId);
      await saveBlobResponse(res, name);
    } catch (err) {
      setLogs([(err as Error).message]);
    }
  };

  const toggleFormat = (f: string) => {
    setFormats((prev) => (prev.includes(f) ? prev.filter((x) => x !== f) : [...prev, f]));
  };

  const updateHub = (key: keyof HubPublishFields, value: string | boolean) => {
    setHub((h) => ({ ...h, [key]: value }));
  };

  return (
    <div>
      <h1 className="page-title">Export &amp; Publish</h1>
      <p className="page-sub">
        Merge LoRA, quantize GGUF, download locally, or publish Seiso-created outputs to Hugging Face.
      </p>

      <div className="card">
        <h3 className="section-title">Export checkpoint</h3>
        <label>Checkpoint path (training output)</label>
        <input value={checkpoint} onChange={(e) => setCheckpoint(e.target.value)} placeholder="~/.seiso/checkpoints/…" />
        <label>Export profile (optional — overrides manual format picks)</label>
        <select value={profile} onChange={(e) => setProfile(e.target.value)}>
          <option value="">Manual formats</option>
          {profiles.map((p) => (
            <option key={p.id} value={p.id}>
              {p.id} ({p.formats.join(", ")})
            </option>
          ))}
        </select>
        <label>Formats</label>
        <div style={{ display: "flex", gap: "0.5rem", marginBottom: "0.75rem", flexWrap: "wrap" }}>
          {["merged", "lora", "full", "gguf"].map((f) => (
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
        <label style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginTop: "0.75rem" }}>
          <input type="checkbox" checked={pushOnExport} onChange={(e) => setPushOnExport(e.target.checked)} />
          Push to Hugging Face when export completes
        </label>
        <button className="btn btn-primary" onClick={start} disabled={busy}>
          Start export
        </button>
        {pushOnExport && (
          <button className="btn" style={{ marginLeft: "0.5rem" }} onClick={precheck} disabled={busy}>
            Precheck Hub
          </button>
        )}
        {lastExportJobId && (
          <button className="btn" style={{ marginLeft: "0.5rem" }} onClick={() => downloadExport(lastExportJobId)}>
            Download GGUF
          </button>
        )}
      </div>

      <div className="card">
        <h3 className="section-title">Publish to Hugging Face</h3>
        <p className="muted-text" style={{ marginBottom: "0.75rem" }}>
          Only Seiso training, export, and RL quant outputs can be published. Provide an API token or use{" "}
          <code>huggingface-cli login</code> / <code>hf auth login</code>.
        </p>
        <div className="settings-grid" style={{ marginBottom: "0.75rem" }}>
          <div>
            <label>HF username</label>
            <input value={hub.username} onChange={(e) => updateHub("username", e.target.value)} placeholder="your-username" />
          </div>
          <div>
            <label>Model name</label>
            <input value={hub.model_name} onChange={(e) => updateHub("model_name", e.target.value)} placeholder="my-finetuned-model" />
          </div>
          <div>
            <label>Author</label>
            <input value={hub.author} onChange={(e) => updateHub("author", e.target.value)} placeholder="Your Name" />
          </div>
          <div>
            <label>License</label>
            <input value={hub.license || ""} onChange={(e) => updateHub("license", e.target.value)} placeholder="apache-2.0" />
          </div>
          <div>
            <label>Base model (optional)</label>
            <input value={hub.base_model || ""} onChange={(e) => updateHub("base_model", e.target.value)} placeholder="meta-llama/Llama-3.2-3B" />
          </div>
        </div>
        <label>Description (model card)</label>
        <textarea
          value={hub.description || ""}
          onChange={(e) => updateHub("description", e.target.value)}
          rows={3}
          placeholder="Fine-tuned on …"
          style={{ width: "100%", marginBottom: "0.75rem" }}
        />
        <label>HF API token (optional if saved in Settings or CLI login)</label>
        <input
          type="password"
          value={hfTokenInput}
          onChange={(e) => setHfTokenInput(e.target.value)}
          placeholder="hf_…"
          autoComplete="off"
        />
        <label style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginTop: "0.5rem" }}>
          <input type="checkbox" checked={!!hub.use_cli} onChange={(e) => updateHub("use_cli", e.target.checked)} />
          Prefer Hugging Face CLI login token
        </label>

        <label style={{ marginTop: "0.75rem" }}>Seiso output to publish</label>
        <select value={selectedModelId} onChange={(e) => { setSelectedModelId(e.target.value); setSelectedExportJobId(""); }}>
          <option value="">— from inventory —</option>
          {publishable.map((m) => (
            <option key={m.id} value={m.id}>
              {m.name} ({m.format}) — {m.source}
            </option>
          ))}
        </select>
        <label>Or completed export job</label>
        <select value={selectedExportJobId} onChange={(e) => { setSelectedExportJobId(e.target.value); setSelectedModelId(""); }}>
          <option value="">— from export job —</option>
          {completedExports.map((j) => (
            <option key={j.id} value={j.id}>
              {j.id.slice(0, 8)} — {j.created_at}
            </option>
          ))}
        </select>
        <button className="btn btn-primary" style={{ marginTop: "0.75rem" }} onClick={publish} disabled={busy}>
          Publish to Hugging Face
        </button>
      </div>

      {publishable.length > 0 && (
        <div className="card">
          <h3 className="section-title">Local outputs</h3>
          <table>
            <thead>
              <tr>
                <th>Name</th>
                <th>Format</th>
                <th>Source</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {publishable.map((m) => (
                <tr key={m.id}>
                  <td>{m.name}</td>
                  <td>{m.format}</td>
                  <td>{m.source}</td>
                  <td>
                    <button className="btn" onClick={() => downloadModel(m.id, m.name)}>
                      Download
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {logs.length > 0 && (
        <div className="card">
          <h3 className="section-title">Log</h3>
          <div className="log-panel">{logs.join("\n")}</div>
        </div>
      )}
    </div>
  );
}
