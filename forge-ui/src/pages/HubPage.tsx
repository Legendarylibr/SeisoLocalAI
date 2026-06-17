import { useCallback, useEffect, useState } from "react";
import { api, CatalogModel, LocalModel } from "@/lib/api";

const FAMILY_LABELS: Record<string, string> = {
  llama: "Llama",
  qwen: "Qwen",
  gemma: "Gemma",
  phi: "Phi",
  mistral: "Mistral",
  deepseek: "DeepSeek",
  llava: "Vision",
  other: "Other",
};

export function HubPage() {
  const [local, setLocal] = useState<LocalModel[]>([]);
  const [catalog, setCatalog] = useState<CatalogModel[]>([]);
  const [families, setFamilies] = useState<string[]>([]);
  const [search, setSearch] = useState("");
  const [family, setFamily] = useState("");
  const [task, setTask] = useState("");
  const [downloading, setDownloading] = useState<string | null>(null);
  const [tab, setTab] = useState<"catalog" | "local">("catalog");
  const [toast, setToast] = useState<string | null>(null);

  const showToast = (msg: string) => {
    setToast(msg);
    setTimeout(() => setToast(null), 4000);
  };

  const refreshLocal = () => api.listModels().then(setLocal).catch(console.error);

  const refreshCatalog = useCallback(() => {
    api
      .catalog(search, family || undefined, task || undefined)
      .then((r) => {
        setCatalog(r.models);
        setFamilies(r.families);
      })
      .catch(console.error);
  }, [search, family, task]);

  useEffect(() => {
    refreshLocal();
  }, []);

  useEffect(() => {
    const t = setTimeout(refreshCatalog, 200);
    return () => clearTimeout(t);
  }, [refreshCatalog]);

  const download = async (repoId: string, variant: "safetensors" | "gguf" = "gguf") => {
    setDownloading(repoId);
    try {
      await api.downloadModel(repoId, undefined, variant);
      await refreshLocal();
      showToast(`Downloaded ${repoId}`);
      setTab("local");
    } catch (e) {
      showToast(e instanceof Error ? e.message : "Download failed");
    } finally {
      setDownloading(null);
    }
  };

  const fmtSize = (n: number) =>
    n > 1e9 ? `${(n / 1e9).toFixed(1)} GB` : n > 1e6 ? `${(n / 1e6).toFixed(1)} MB` : `${n} B`;

  return (
    <div>
      <h1 className="page-title">Model Hub</h1>
      <p className="page-sub">Browse 40+ popular Hugging Face models. Download for chat (GGUF) or training (safetensors).</p>

      {toast && <div className="toast">{toast}</div>}

      <div className="tabs">
        <button className={`tab ${tab === "catalog" ? "active" : ""}`} onClick={() => setTab("catalog")}>
          Catalog ({catalog.length})
        </button>
        <button className={`tab ${tab === "local" ? "active" : ""}`} onClick={() => setTab("local")}>
          Local ({local.length})
        </button>
      </div>

      {tab === "catalog" && (
        <>
          <div className="card filters">
            <input
              className="search-input"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search models… (Llama, Qwen, DeepSeek, code…)"
            />
            <div className="filter-row">
              <select value={family} onChange={(e) => setFamily(e.target.value)}>
                <option value="">All families</option>
                {families.map((f) => (
                  <option key={f} value={f}>{FAMILY_LABELS[f] || f}</option>
                ))}
              </select>
              <select value={task} onChange={(e) => setTask(e.target.value)}>
                <option value="">All tasks</option>
                <option value="chat">Chat</option>
                <option value="code">Code</option>
                <option value="vision">Vision</option>
                <option value="embedding">Embedding</option>
              </select>
            </div>
          </div>

          <div className="model-grid">
            {catalog.map((m) => (
              <div key={m.repo_id} className="model-card">
                <div className="model-card-header">
                  <span className={`family-tag family-${m.family}`}>{FAMILY_LABELS[m.family] || m.family}</span>
                  <span className="params-tag">{m.params}</span>
                </div>
                <h3 className="model-name">{m.name}</h3>
                <p className="model-repo">{m.repo_id}</p>
                <div className="model-tags">
                  <span className="badge">{m.task}</span>
                  {m.tags.map((t) => (
                    <span key={t} className="badge badge-dim">{t}</span>
                  ))}
                </div>
                <div className="model-actions">
                  <button
                    className="btn btn-primary"
                    disabled={downloading === m.repo_id}
                    onClick={() => download(m.repo_id, "gguf")}
                  >
                    {downloading === m.repo_id ? "…" : "Chat (GGUF)"}
                  </button>
                  <button
                    className="btn"
                    disabled={downloading === m.repo_id}
                    onClick={() => download(m.repo_id, "safetensors")}
                  >
                    Train
                  </button>
                </div>
              </div>
            ))}
          </div>
        </>
      )}

      {tab === "local" && (
        <div className="card">
          {local.length === 0 ? (
            <div className="empty-state">
              <p>No local models yet.</p>
              <button className="btn btn-primary" onClick={() => setTab("catalog")}>Browse catalog</button>
            </div>
          ) : (
            <table>
              <thead>
                <tr><th>Name</th><th>Format</th><th>Source</th><th>Size</th></tr>
              </thead>
              <tbody>
                {local.map((m) => (
                  <tr key={m.id}>
                    <td>{m.name}</td>
                    <td><span className="badge">{m.format || "—"}</span></td>
                    <td className="muted-cell">{m.source}</td>
                    <td>{fmtSize(m.size_bytes)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}
    </div>
  );
}
