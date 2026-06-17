import { useCallback, useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, CatalogModel, HardwareSummary, LocalModel } from "@/lib/api";
import { chatPath, chatPathForLocalModel } from "@/lib/chatModel";
import { HardwareFitBadge } from "@/components/HardwareFitBadge";
import { PageHeader } from "@/components/PageHeader";
import { IconClose, IconSearch } from "@/components/Icons";

const FAMILY_LABELS: Record<string, string> = {
  llama: "Llama",
  qwen: "Qwen",
  gemma: "Gemma",
  phi: "Phi",
  mistral: "Mistral",
  deepseek: "DeepSeek",
  kimi: "Kimi",
  minimax: "MiniMax",
  nemotron: "Nemotron",
  glm: "GLM",
  ibm: "IBM Granite",
  olmo: "Olmo",
  llava: "Vision",
  other: "Other",
};

const QUICK_FILTERS = [
  { label: "Fits your GPU", task: "", q: "", fitsOnly: true },
  { label: "Featured", task: "", q: "new", fitsOnly: false },
  { label: "Chat", task: "chat", q: "", fitsOnly: false },
  { label: "Code", task: "code", q: "", fitsOnly: false },
  { label: "Llama", task: "", q: "llama", fitsOnly: false },
  { label: "Gemma 4", task: "", q: "gemma4", fitsOnly: false },
  { label: "Qwen 3.6", task: "", q: "qwen3.6", fitsOnly: false },
  { label: "DeepSeek", task: "", q: "deepseek", fitsOnly: false },
  { label: "Mistral", task: "", q: "mistral", fitsOnly: false },
  { label: "Kimi", task: "", q: "kimi", fitsOnly: false },
  { label: "Phi", task: "", q: "phi", fitsOnly: false },
  { label: "GLM", task: "", q: "glm", fitsOnly: false },
] as const;

export function HubPage() {
  const navigate = useNavigate();
  const [local, setLocal] = useState<LocalModel[]>([]);
  const [catalog, setCatalog] = useState<CatalogModel[]>([]);
  const [families, setFamilies] = useState<string[]>([]);
  const [total, setTotal] = useState(0);
  const [search, setSearch] = useState("");
  const [family, setFamily] = useState("");
  const [task, setTask] = useState("");
  const [fitsOnly, setFitsOnly] = useState(false);
  const [hwSummary, setHwSummary] = useState<HardwareSummary | null>(null);
  const [downloading, setDownloading] = useState<string | null>(null);
  const [downloadAction, setDownloadAction] = useState<"chat" | "train" | null>(null);
  const [tab, setTab] = useState<"catalog" | "local">("catalog");
  const [toast, setToast] = useState<string | null>(null);
  const [hubReady, setHubReady] = useState<boolean | null>(null);
  const [hubError, setHubError] = useState<string | null>(null);

  const showToast = (msg: string) => {
    setToast(msg);
    setTimeout(() => setToast(null), 4000);
  };

  const refreshLocal = () => api.listModels().then(setLocal).catch(console.error);

  const refreshCatalog = useCallback(() => {
    api
      .catalog(search, family || undefined, task || undefined, fitsOnly)
      .then((r) => {
        setCatalog(r.models);
        setFamilies(r.families);
        setTotal(r.total);
        if (r.hardware_summary) setHwSummary(r.hardware_summary);
      })
      .catch(console.error);
  }, [search, family, task, fitsOnly]);

  useEffect(() => {
    refreshLocal();
    api.hfStatus().then((s) => {
      setHubReady(s.ready_for_download);
      setHubError(s.connectivity.error);
    }).catch(() => setHubReady(null));
  }, []);

  useEffect(() => {
    const t = setTimeout(refreshCatalog, 180);
    return () => clearTimeout(t);
  }, [refreshCatalog]);

  const openChat = (repoId: string, downloadBytes?: number) => {
    navigate(chatPath({ repo: repoId, downloadBytes }));
  };

  const openTrain = async (repoId: string) => {
    setDownloading(repoId);
    setDownloadAction("train");
    try {
      await api.downloadModel(repoId, undefined, "safetensors");
      await refreshLocal();
      showToast(`Model cached for training`);
      navigate(`/train?model=${encodeURIComponent(repoId)}`);
    } catch (e) {
      showToast(e instanceof Error ? e.message : "Download failed");
    } finally {
      setDownloading(null);
      setDownloadAction(null);
    }
  };

  const applyQuickFilter = (q: string, t: string, fits: boolean) => {
    setSearch(q);
    setTask(t);
    setFamily("");
    setFitsOnly(fits);
  };

  const activeFilters = useMemo(() => {
    const parts: string[] = [];
    if (search) parts.push(`"${search}"`);
    if (family) parts.push(FAMILY_LABELS[family] || family);
    if (task) parts.push(task);
    return parts;
  }, [search, family, task]);

  const fmtSize = (n: number) => {
    const gib = 1024 ** 3;
    const mib = 1024 ** 2;
    return n >= gib ? `${(n / gib).toFixed(1)} GB` : n >= mib ? `${(n / mib).toFixed(1)} MB` : `${n} B`;
  };

  return (
    <div className="hub-page">
      <PageHeader
        title="Model Hub"
        subtitle="Newest models first, ranked for your hardware when possible. Detection stays on this machine — nothing is uploaded."
      />

      {hwSummary && (
        <div className="hw-inline-banner card">
          <span className="trust-badge">{hwSummary.tier_label}</span>
          <span className="muted-text">
            ~{Math.round(hwSummary.vram_headroom_mb / 1024)} GB {hwSummary.memory_headroom_label || "memory"} free · prefers {hwSummary.preferred_inference_backend_label || hwSummary.preferred_inference_backend}
          </span>
        </div>
      )}

      {hubReady === false && (
        <div className="card" style={{ borderColor: "var(--warn, #c9a227)", marginBottom: "1rem" }}>
          <strong>Hugging Face Hub not ready</strong>
          <p className="muted-text" style={{ marginTop: "0.35rem" }}>
            {hubError || "Check network connectivity and install dependencies from Settings."}
          </p>
        </div>
      )}

      {toast && <div className="toast">{toast}</div>}

      <div className="tabs">
        <button className={`tab ${tab === "catalog" ? "active" : ""}`} onClick={() => setTab("catalog")}>
          Catalog ({total})
        </button>
        <button className={`tab ${tab === "local" ? "active" : ""}`} onClick={() => setTab("local")}>
          Local ({local.length})
        </button>
      </div>

      {tab === "catalog" && (
        <>
          <div className="card filters hub-filters">
            <div className="hub-search-wrap">
              <span className="hub-search-icon" aria-hidden>
                <IconSearch size={16} />
              </span>
              <input
                className="hub-search-input"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="Search by name, family, size (e.g. qwen 7b coder)…"
              />
              {search && (
                <button type="button" className="hub-search-clear" onClick={() => setSearch("")} aria-label="Clear">
                  <IconClose size={14} />
                </button>
              )}
            </div>
            <div className="hub-quick-filters">
              {QUICK_FILTERS.map((f) => (
                <button
                  key={f.label}
                  type="button"
                  className={`hub-chip${search === f.q && task === f.task && fitsOnly === f.fitsOnly ? " active" : ""}`}
                  onClick={() => applyQuickFilter(f.q, f.task, f.fitsOnly)}
                >
                  {f.label}
                </button>
              ))}
            </div>
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
            <p className="hub-results-meta">
              {total} model{total === 1 ? "" : "s"}
              {activeFilters.length > 0 && <> · filtered by {activeFilters.join(" · ")}</>}
            </p>
          </div>

          {catalog.length === 0 ? (
            <div className="card empty-state">
              <p>No models match your search.</p>
              <button
                type="button"
                className="btn"
                onClick={() => {
                  setSearch("");
                  setFamily("");
                  setTask("");
                  setFitsOnly(false);
                }}
              >
                Clear filters
              </button>
            </div>
          ) : (
            <div className="model-grid">
              {catalog.map((m) => (
                <div key={m.repo_id} className={`model-card${m.featured ? " model-card-featured" : ""}`}>
                  <div className="model-card-header">
                    <span className={`family-tag family-${m.family}`}>{FAMILY_LABELS[m.family] || m.family}</span>
                    <span className="params-tag">{m.params}</span>
                    {m.featured && <span className="badge badge-featured">Featured</span>}
                    <HardwareFitBadge fit={m.hardware_fit} label={m.hardware_fit_label} />
                  </div>
                  <h3 className="model-name">{m.name}</h3>
                  <p className="model-repo">{m.repo_id}</p>
                  {m.download_bytes ? (
                    <p className="model-download-size muted-text">
                      {m.download_bytes_estimated ? "~" : ""}{fmtSize(m.download_bytes)} download · {m.quant}
                      {m.gguf_repo && m.gguf_repo !== m.repo_id ? ` · via ${m.gguf_repo.split("/").pop()}` : ""}
                    </p>
                  ) : null}
                  {m.hardware_note && <p className="model-hw-note">{m.hardware_note}</p>}
                  <div className="model-tags">
                    <span className="badge">{m.task}</span>
                    {m.tags.map((t) => (
                      <span key={t} className="badge badge-dim">{t}</span>
                    ))}
                  </div>
                  <div className="model-actions">
                    <button
                      className="btn btn-primary"
                      disabled={downloading === m.repo_id && downloadAction === "train"}
                      onClick={() => openChat(m.repo_id, m.download_bytes)}
                    >
                      Download and chat
                    </button>
                    <button
                      className="btn"
                      disabled={downloading === m.repo_id}
                      onClick={() => openTrain(m.repo_id)}
                    >
                      {downloading === m.repo_id && downloadAction === "train" ? "Caching…" : "Train/Finetune"}
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </>
      )}

      {tab === "local" && (
        <div className="card">
          {local.length === 0 ? (
            <div className="empty-state">
              <p>No local models yet.</p>
              <button className="btn btn-primary" type="button" onClick={() => setTab("catalog")}>
                Browse catalog
              </button>
            </div>
          ) : (
            <table>
              <thead>
                <tr><th>Name</th><th>Format</th><th>Source</th><th>Size</th><th></th></tr>
              </thead>
              <tbody>
                {local.map((m) => (
                  <tr key={m.id}>
                    <td>{m.name}</td>
                    <td><span className="badge">{m.format || "—"}</span></td>
                    <td className="muted-cell">{m.source}</td>
                    <td>{fmtSize(m.size_bytes)}</td>
                    <td>
                      {m.format === "gguf" && (
                        <button
                          type="button"
                          className="btn"
                          style={{ padding: "0.25rem 0.5rem", fontSize: "0.78rem" }}
                          onClick={() => navigate(chatPathForLocalModel(m))}
                        >
                          Chat
                        </button>
                      )}
                      {m.source?.startsWith("hf:") && m.format === "safetensors" && (
                        <button
                          type="button"
                          className="btn"
                          style={{ padding: "0.25rem 0.5rem", fontSize: "0.78rem" }}
                          onClick={() => navigate(`/train?model=${encodeURIComponent(m.source!.slice(3))}`)}
                        >
                          Train
                        </button>
                      )}
                    </td>
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
