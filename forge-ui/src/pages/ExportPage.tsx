import { useEffect, useRef, useState } from "react";
import { api, ExportJob, HubPublishFields, PublishableModel, RLQuantJob, subscribeSSE } from "@/lib/api";
import { invalidateApiCache } from "@/lib/api/getCache";
import { appendBoundedLog } from "@/lib/api/sse";
import { DataTable } from "@/components/research/DataTable";
import { FormSection } from "@/components/research/FormSection";
import { LogStream } from "@/components/research/LogStream";
import { StudioCardBody } from "@/components/studio/StudioCardBody";
import { StudioCardHeader } from "@/components/studio/StudioCardHeader";
import { StudioPageShell } from "@/components/StudioPageShell";

function emptyHub(): HubPublishFields {
  return { username: "", model_name: "", author: "", license: "apache-2.0", description: "", use_cli: false };
}

async function saveBlobResponse(res: Response, fallbackName: string) {
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    const detail = Array.isArray(err.detail) ? err.detail.map((d: unknown) => JSON.stringify(d)).join("; ") : err.detail;
    throw new Error(detail || "Download failed");
  }
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
  const [ggufQuants, setGgufQuants] = useState<string[]>(["q4_k_m"]);
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
  const streamAbortRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    api.listRLQuantJobs().then(setRlJobs).catch(console.error);
    api.listPublishableOutputs().then(setPublishable).catch(console.error);
    api.listExportJobs().then(setExportJobs).catch(console.error);
    api.listExportProfiles().then(setProfiles).catch(console.error);
    return () => streamAbortRef.current?.();
  }, []);

  const completedRlJobs = rlJobs.filter((j) => j.status === "completed" && j.gguf_quants?.length);
  const completedExports = exportJobs.filter((j) => j.status === "completed");
  const GGUF_QUANT_OPTIONS = ["q2_k", "q3_k_m", "q4_k_m", "q5_k_m", "q6_k", "q8_0", "f16"];

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
        rlQuantJobId ? undefined : ggufQuants,
      );
      setLastExportJobId(res.job_id);
      streamAbortRef.current?.();
      streamAbortRef.current = subscribeSSE(`/export/jobs/${res.job_id}/stream`, (event, data) => {
        if (event === "log" || event === "result") {
          setLogs((l) => {
            const next = appendBoundedLog(l, data);
            return next;
          });
        }
        if (event === "error") setLogs((l) => appendBoundedLog(l, `ERROR: ${data}`));
        if (event === "result") {
          invalidateApiCache("/inference/models");
          invalidateApiCache("/training/models");
        }
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
        gguf_quantizations: rlQuantJobId ? undefined : ggufQuants,
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

  const toggleQuant = (quant: string) => {
    setGgufQuants((prev) => (prev.includes(quant) ? prev.filter((q) => q !== quant) : [...prev, quant]));
  };

  const updateHub = (key: keyof HubPublishFields, value: string | boolean) => {
    setHub((h) => ({ ...h, [key]: value }));
  };

  return (
    <StudioPageShell
      title="Export & Publish"
      subtitle="Merge LoRA, quantize GGUF, download locally, or publish Seiso-created outputs to Hugging Face."
    >
      <div className="train-layout train-layout--studio train-layout--export-fit">
        <div className="card studio-card">
          <StudioCardHeader
            icon="①"
            title="Export checkpoint"
            description="Merge LoRA, pick formats, and optionally push to Hub when done."
          />
          <StudioCardBody>
          <FormSection title="Source" hint="Training output directory to export from.">
            <div className="form-field">
              <label>Checkpoint path</label>
              <input value={checkpoint} onChange={(e) => setCheckpoint(e.target.value)} placeholder="~/.seiso/checkpoints/…" />
            </div>
          </FormSection>
          <FormSection title="Formats & profile" hint="Manual picks or a named export profile." collapsible defaultOpen>
            <div className="option-grid">
              <div className="form-field">
                <label>Export profile (optional)</label>
                <select value={profile} onChange={(e) => setProfile(e.target.value)}>
                  <option value="">Manual formats</option>
                  {profiles.map((p) => (
                    <option key={p.id} value={p.id}>
                      {p.id} ({p.formats.join(", ")})
                    </option>
                  ))}
                </select>
              </div>
              <div className="form-field">
                <label>RL quant job (optional)</label>
                <select value={rlQuantJobId} onChange={(e) => setRlQuantJobId(e.target.value)}>
                  <option value="">Manual / default quants</option>
                  {completedRlJobs.map((j) => (
                    <option key={j.id} value={j.id}>
                      {j.id.slice(0, 8)} — {j.gguf_quants.join(", ")}
                    </option>
                  ))}
                </select>
              </div>
            </div>
            <div className="form-field">
              <label>Formats</label>
              <div className="studio-chip-group">
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
            </div>
            {formats.includes("gguf") && !rlQuantJobId && (
              <div className="form-field">
                <label>GGUF quantizations</label>
                <div className="studio-chip-group">
                  {GGUF_QUANT_OPTIONS.map((q) => (
                    <button
                      key={q}
                      type="button"
                      className={`btn ${ggufQuants.includes(q) ? "btn-primary" : ""}`}
                      onClick={() => toggleQuant(q)}
                    >
                      {q}
                    </button>
                  ))}
                </div>
              </div>
            )}
            <label className="studio-checkbox-item studio-checkbox-item-standalone">
              <input type="checkbox" checked={pushOnExport} onChange={(e) => setPushOnExport(e.target.checked)} />
              Push to Hugging Face when export completes
            </label>
          </FormSection>
          </StudioCardBody>
          <div className="studio-action-bar studio-action-bar-flush">
            <button className="btn btn-primary btn-lg" onClick={start} disabled={busy}>
              Start export
            </button>
            {pushOnExport && (
              <button className="btn" onClick={precheck} disabled={busy}>
                Precheck Hub
              </button>
            )}
            {lastExportJobId && (
              <button className="btn" onClick={() => downloadExport(lastExportJobId)}>
                Download GGUF
              </button>
            )}
          </div>
        </div>

        <div className="card studio-card">
          <StudioCardHeader
            icon="②"
            title="Publish to Hugging Face"
            description="Upload Seiso-created outputs with a model card."
          />
          <StudioCardBody>
          <p className="field-hint">
            Only Seiso training, export, and RL quant outputs can be published. Provide an API token or use{" "}
            <code>huggingface-cli login</code> / <code>hf auth login</code>.
          </p>
          <FormSection title="Hub metadata" hint="Repository identity and model card fields." collapsible defaultOpen>
            <div className="option-grid">
              <div className="form-field">
                <label>HF username</label>
                <input value={hub.username} onChange={(e) => updateHub("username", e.target.value)} placeholder="your-username" />
              </div>
              <div className="form-field">
                <label>Model name</label>
                <input value={hub.model_name} onChange={(e) => updateHub("model_name", e.target.value)} placeholder="my-finetuned-model" />
              </div>
              <div className="form-field">
                <label>Author</label>
                <input value={hub.author} onChange={(e) => updateHub("author", e.target.value)} placeholder="Your Name" />
              </div>
              <div className="form-field">
                <label>License</label>
                <input value={hub.license || ""} onChange={(e) => updateHub("license", e.target.value)} placeholder="apache-2.0" />
              </div>
              <div className="form-field">
                <label>Base model (optional)</label>
                <input value={hub.base_model || ""} onChange={(e) => updateHub("base_model", e.target.value)} placeholder="meta-llama/Llama-3.2-3B" />
              </div>
              <div className="form-field">
                <label>HF API token (optional)</label>
                <input
                  type="password"
                  value={hfTokenInput}
                  onChange={(e) => setHfTokenInput(e.target.value)}
                  placeholder="hf_…"
                  autoComplete="off"
                />
              </div>
            </div>
            <div className="form-field">
              <label>Description (model card)</label>
              <textarea
                value={hub.description || ""}
                onChange={(e) => updateHub("description", e.target.value)}
                rows={2}
                placeholder="Fine-tuned on …"
              />
            </div>
            <label className="studio-checkbox-item studio-checkbox-item-standalone">
              <input type="checkbox" checked={!!hub.use_cli} onChange={(e) => updateHub("use_cli", e.target.checked)} />
              Prefer Hugging Face CLI login token
            </label>
          </FormSection>
          <FormSection title="Output selection" hint="Pick inventory or a completed export job." collapsible defaultOpen={false}>
            <div className="option-grid">
              <div className="form-field">
                <label>Seiso output to publish</label>
                <select value={selectedModelId} onChange={(e) => { setSelectedModelId(e.target.value); setSelectedExportJobId(""); }}>
                  <option value="">— from inventory —</option>
                  {publishable.map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.name} ({m.format}) — {m.source}
                    </option>
                  ))}
                </select>
              </div>
              <div className="form-field">
                <label>Or completed export job</label>
                <select value={selectedExportJobId} onChange={(e) => { setSelectedExportJobId(e.target.value); setSelectedModelId(""); }}>
                  <option value="">— from export job —</option>
                  {completedExports.map((j) => (
                    <option key={j.id} value={j.id}>
                      {j.id.slice(0, 8)} — {j.created_at}
                    </option>
                  ))}
                </select>
              </div>
            </div>
          </FormSection>
          </StudioCardBody>
          <div className="studio-action-bar studio-action-bar-flush">
            <button className="btn btn-primary btn-lg" onClick={publish} disabled={busy}>
              Publish to Hugging Face
            </button>
          </div>
        </div>

        <div className="card studio-card studio-card-scroll">
          <StudioCardHeader
            icon="③"
            title="Activity"
            description="Local inventory downloads and export/publish logs."
            tone="monitor"
          />
          {publishable.length > 0 ? (
            <DataTable
              columns={[
                { key: "name", header: "Name" },
                { key: "format", header: "Format" },
                { key: "source", header: "Source" },
                {
                  key: "actions",
                  header: "",
                  render: (m) => (
                    <button className="btn" onClick={() => downloadModel(m.id, m.name)}>
                      Download
                    </button>
                  ),
                },
              ]}
              rows={publishable.slice(0, 6)}
              getRowKey={(m) => m.id}
              emptyMessage="No local outputs yet."
            />
          ) : (
            <p className="research-empty">No local outputs yet.</p>
          )}
          <LogStream
            logs={logs}
            emptyMessage="Export and publish logs appear here."
            fill
            label="Export log"
          />
        </div>
      </div>
    </StudioPageShell>
  );
}
