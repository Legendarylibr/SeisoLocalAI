import { useCallback, useEffect, useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api, CatalogModel, HardwareSummary, LocalModel } from "@/lib/api";
import { usePlatformSettings } from "@/context/PlatformSettingsContext";
import { chatPath, chatPathForLocalModel } from "@/lib/chatModel";
import { trainPath } from "@/lib/hubDownload";
import { HardwareFitBadge } from "@/components/HardwareFitBadge";
import { ModelCardSkeleton } from "@/components/ModelCardSkeleton";
import { PageHeader } from "@/components/PageHeader";
import { StatusCallout } from "@/components/StatusCallout";
import { Tabs } from "@/components/Tabs";
import { IconClose, IconGlobe, IconHardDrive, IconSearch } from "@/components/Icons";

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

const TASK_LABELS: Record<string, string> = {
  chat: "Chat",
  code: "Code",
  vision: "Vision",
  embedding: "Embedding",
};

export function HubPage() {
  const navigate = useNavigate();
  const { hfStatus } = usePlatformSettings();
  const hubReady = hfStatus ? hfStatus.ready_for_download : null;
  const hubError = hfStatus
    ? hfStatus.connectivity.error || hfStatus.connectivity.warning || null
    : null;
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
  const [catalogLoading, setCatalogLoading] = useState(true);

  const refreshLocal = () => api.listModels().then(setLocal).catch(console.error);

  const refreshCatalog = useCallback(() => {
    setCatalogLoading(true);
    api
      .catalog(search, family || undefined, task || undefined, fitsOnly)
      .then((r) => {
        setCatalog(r.models);
        setFamilies(r.families);
        setTotal(r.total);
        if (r.hardware_summary) setHwSummary(r.hardware_summary);
      })
      .catch(console.error)
      .finally(() => setCatalogLoading(false));
  }, [search, family, task, fitsOnly]);

  useEffect(() => {
    refreshLocal();
    return () => {
      setDownloading(null);
      setDownloadAction(null);
    };
  }, []);

  useEffect(() => {
    const t = setTimeout(refreshCatalog, 180);
    return () => clearTimeout(t);
  }, [refreshCatalog]);

  const openChat = (repoId: string, downloadBytes?: number) => {
    setDownloading(repoId);
    setDownloadAction("chat");
    navigate(chatPath({ repo: repoId, downloadBytes }));
  };

  const openTrain = (repoId: string, downloadBytes?: number) => {
    setDownloading(repoId);
    setDownloadAction("train");
    navigate(trainPath(repoId, downloadBytes));
  };

  const applyQuickFilter = (q: string, t: string, fits: boolean) => {
    setSearch(q);
    setTask(t);
    setFamily("");
    setFitsOnly(fits);
  };

  const activeFilters = useMemo(() => {
    const parts: string[] = [];
    if (fitsOnly) parts.push("fits your hardware");
    if (search) parts.push(`"${search}"`);
    if (family) parts.push(FAMILY_LABELS[family] || family);
    if (task) parts.push(TASK_LABELS[task] || task);
    return parts;
  }, [search, family, task, fitsOnly]);

  const fmtSize = (n: number) => {
    const gib = 1024 ** 3;
    const mib = 1024 ** 2;
    return n >= gib ? `${(n / gib).toFixed(1)} GB` : n >= mib ? `${(n / mib).toFixed(1)} MB` : `${n} B`;
  };

  return (
    <div className="hub-page">
      <PageHeader
        title="Model Hub"
        subtitle="Download GGUF models to Seiso's local cache for llama.cpp, or use separately pulled Ollama models from Chat. Detection stays on this machine."
        group="Models"
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
        <StatusCallout
          tone="warn"
          title="Hugging Face Hub not ready"
          action={
            <Link to="/settings?tab=huggingface" className="btn btn-sm">
              Open settings
            </Link>
          }
        >
          {hubError || "Check network connectivity and install dependencies from Settings."}
        </StatusCallout>
      )}
      {hubReady === true && (
        <StatusCallout tone="success" title="Public downloads ready">
          No token needed for public GGUF models. Add a Hugging Face token in Settings only for gated repos or publishing.
        </StatusCallout>
      )}

      <Tabs
        className="hub-tab-bar"
        aria-label="Model sources"
        value={tab}
        onChange={setTab}
        items={[
          {
            id: "catalog",
            label: "Catalog",
            description: "Browse Hugging Face — ranked for your GPU",
            icon: <IconGlobe size={16} />,
            count: total,
          },
          {
            id: "local",
            label: "Local library",
            description: "Models downloaded and ready on disk",
            icon: <IconHardDrive size={16} />,
            count: local.length,
          },
        ]}
      />

      {tab === "catalog" && (
        <div className="hub-tab-panel">
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
              {catalogLoading ? "Searching catalog…" : (
                <>
                  {total} model{total === 1 ? "" : "s"}
                  {activeFilters.length > 0 && <> · filtered by {activeFilters.join(" · ")}</>}
                </>
              )}
            </p>
          </div>

          {catalogLoading && catalog.length === 0 ? (
            <div className="model-grid" aria-busy="true" aria-label="Loading models">
              {Array.from({ length: 6 }, (_, i) => (
                <ModelCardSkeleton key={i} />
              ))}
            </div>
          ) : catalog.length === 0 ? (
            <div className="card empty-state">
              <div className="empty-state-icon" aria-hidden>
                <IconSearch size={28} />
              </div>
              <p className="empty-state-title">No models match your search</p>
              <p className="empty-state-desc">Try clearing filters or browsing a different family.</p>
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
              {catalog.map((m) => {
                const embeddingOnly = m.download_available === false;
                const mirrorUnverified = m.download_mirror_verified === false && !!m.download_error;
                return (
                <div
                  key={m.repo_id}
                  className={`model-card${m.featured ? " model-card-featured" : ""}${embeddingOnly ? " model-card-unavailable" : ""}`}
                >
                  <div className="model-card-header">
                    <span className={`family-tag family-${m.family}`}>{FAMILY_LABELS[m.family] || m.family}</span>
                    {m.task && <span className="task-tag">{TASK_LABELS[m.task] || m.task}</span>}
                    <span className="params-tag">{m.params}</span>
                    {m.featured && <span className="badge badge-featured">Featured</span>}
                    <HardwareFitBadge fit={m.hardware_fit} label={m.hardware_fit_label} />
                  </div>
                  <h3 className="model-name">{m.name}</h3>
                  <p className="model-repo" title={m.repo_id}>{m.repo_id}</p>
                  {m.download_bytes ? (
                    <p className="model-download-size muted-text">
                      <span className="model-meta-pill">GGUF</span>
                      {m.download_bytes_estimated ? "~" : ""}{fmtSize(m.download_bytes)}
                      <span className="model-meta-sep">·</span>
                      {m.quant}
                      {m.gguf_repo && m.gguf_repo !== m.repo_id ? (
                        <>
                          <span className="model-meta-sep">·</span>
                          via {m.gguf_repo.split("/").pop()}
                        </>
                      ) : null}
                    </p>
                  ) : null}
                  {m.hardware_note && !embeddingOnly && <p className="model-hw-note">{m.hardware_note}</p>}
                  {mirrorUnverified && (
                    <p className="model-mirror-warn">
                      Mirror not pre-verified: {m.download_error} You can still try downloading.
                    </p>
                  )}
                  {embeddingOnly && (
                    <p className="model-unavailable-note">
                      Embedding models are not supported for direct chat download.
                    </p>
                  )}
                  <div className="model-actions">
                    <button
                      className="btn btn-primary"
                      disabled={downloading === m.repo_id || embeddingOnly}
                      onClick={() => openChat(m.repo_id, m.download_bytes)}
                      title="Download public GGUF to local cache and open chat"
                    >
                      {downloading === m.repo_id && downloadAction === "chat" ? "Opening…" : "Chat with GGUF"}
                    </button>
                    <button
                      className="btn"
                      disabled={downloading === m.repo_id || embeddingOnly}
                      onClick={() => openTrain(m.repo_id, m.download_bytes)}
                      title="Download safetensors snapshot and open Training Studio"
                    >
                      {downloading === m.repo_id && downloadAction === "train" ? "Opening…" : "Fine-tune"}
                    </button>
                  </div>
                </div>
              );
              })}
            </div>
          )}
        </div>
      )}

      {tab === "local" && (
        <div className="hub-tab-panel">
          <div className="card">
            <div className="card-head">
              <span className="card-head-icon" aria-hidden>
                <IconHardDrive size={18} />
              </span>
              <div className="card-head-text">
                <h3>Local models</h3>
                <p>Cached weights on this machine — chat with GGUF or fine-tune safetensors snapshots.</p>
              </div>
            </div>
          {local.length === 0 ? (
            <div className="empty-state">
              <p>No local models yet.</p>
              <button className="btn btn-primary" type="button" onClick={() => setTab("catalog")}>
                Browse catalog
              </button>
            </div>
          ) : (
            <div className="local-models-grid">
              {local.map((m) => (
                <div key={m.id} className="local-model-row">
                  <div className="local-model-main">
                    <div className="local-model-name">{m.name}</div>
                    <div className="local-model-meta">
                      <span className="badge">{m.format || "—"}</span>
                      <span>{m.source}</span>
                      <span>{fmtSize(m.size_bytes)}</span>
                    </div>
                  </div>
                  <div className="local-model-actions">
                      {m.format === "gguf" && (
                        <button
                          type="button"
                          className="btn btn-primary"
                          onClick={() => navigate(chatPathForLocalModel(m))}
                        >
                          Chat
                        </button>
                      )}
                      {m.source?.startsWith("hf:") && m.format === "safetensors" && (
                        <button
                          type="button"
                          className="btn"
                          onClick={() => navigate(`/train?model=${encodeURIComponent(m.source!.slice(3))}`)}
                        >
                          Train
                        </button>
                      )}
                  </div>
                </div>
              ))}
            </div>
          )}
          </div>
        </div>
      )}
    </div>
  );
}
