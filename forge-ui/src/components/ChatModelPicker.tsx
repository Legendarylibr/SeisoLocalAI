import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, CatalogModel, InferenceModelOption } from "@/lib/api";
import { formatBytes } from "@/lib/modelProgress";
import { IconChevronDown, IconClose, IconSearch } from "@/components/Icons";

type ChatModelPickerProps = {
  models: InferenceModelOption[];
  selection: string;
  disabled?: boolean;
  switching?: boolean;
  modelLabel: (m: InferenceModelOption) => string;
  onSelectLocal: (modelId: string) => void | Promise<void>;
  onSelectCatalog: (repoId: string) => void | Promise<void>;
};

export function ChatModelPicker({
  models,
  selection,
  disabled,
  switching,
  modelLabel,
  onSelectLocal,
  onSelectCatalog,
}: ChatModelPickerProps) {
  const [open, setOpen] = useState(false);
  const [search, setSearch] = useState("");
  const [catalog, setCatalog] = useState<CatalogModel[]>([]);
  const [catalogLoading, setCatalogLoading] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);
  const searchRef = useRef<HTMLInputElement>(null);

  const selected = useMemo(() => models.find((m) => m.id === selection) ?? null, [models, selection]);

  const downloadedRepos = useMemo(
    () =>
      new Set(
        models
          .map((m) => (m.source?.startsWith("hf:") ? m.source.slice(3) : null))
          .filter((r): r is string => Boolean(r)),
      ),
    [models],
  );

  const refreshCatalog = useCallback((q: string) => {
    setCatalogLoading(true);
    api
      .catalog(q, undefined, undefined, false)
      .then((r) => setCatalog(r.models))
      .catch(() => setCatalog([]))
      .finally(() => setCatalogLoading(false));
  }, []);

  useEffect(() => {
    if (!open) return;
    const t = setTimeout(() => refreshCatalog(search), search ? 180 : 0);
    return () => clearTimeout(t);
  }, [open, search, refreshCatalog]);

  useEffect(() => {
    if (!open) return;
    const onDoc = (e: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, [open]);

  useEffect(() => {
    if (open) {
      setSearch("");
      requestAnimationFrame(() => searchRef.current?.focus());
    }
  }, [open]);

  const q = search.toLowerCase().trim();
  const filteredLocal = useMemo(() => {
    if (!q) return models;
    return models.filter(
      (m) =>
        m.name.toLowerCase().includes(q) ||
        m.source.toLowerCase().includes(q) ||
        m.source_label.toLowerCase().includes(q),
    );
  }, [models, q]);

  const hubModels = useMemo(
    () => catalog.filter((c) => !downloadedRepos.has(c.repo_id)),
    [catalog, downloadedRepos],
  );

  const triggerLabel = switching
    ? "Loading model…"
    : selected
      ? modelLabel(selected)
      : models.length === 0
        ? "Select a model…"
        : "Select a model…";

  const pickLocal = (modelId: string) => {
    setOpen(false);
    void onSelectLocal(modelId);
  };

  const pickCatalog = (repoId: string) => {
    setOpen(false);
    void onSelectCatalog(repoId);
  };

  const empty =
    !catalogLoading && filteredLocal.length === 0 && hubModels.length === 0;

  return (
    <div className="chat-model-picker" ref={rootRef}>
      <button
        type="button"
        className="chat-model-picker-trigger"
        onClick={() => !disabled && setOpen((v) => !v)}
        disabled={disabled}
        aria-haspopup="listbox"
        aria-expanded={open}
        title={disabled ? "Switch to Local to pick an on-device model" : "Switch model — unloads previous from VRAM"}
      >
        <span className="chat-model-picker-label">{triggerLabel}</span>
        <span className="chat-model-picker-chevron" aria-hidden>
          <IconChevronDown size={14} />
        </span>
      </button>

      {open && (
        <div className="chat-model-picker-menu" role="listbox">
          <div className="chat-model-picker-search">
            <span className="chat-model-picker-search-icon" aria-hidden>
              <IconSearch size={15} />
            </span>
            <input
              ref={searchRef}
              type="search"
              className="chat-model-picker-search-input"
              placeholder="Search local & Hugging Face models…"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Escape") {
                  e.preventDefault();
                  setOpen(false);
                }
              }}
            />
            {search && (
              <button
                type="button"
                className="chat-model-picker-search-clear"
                onClick={() => setSearch("")}
                aria-label="Clear search"
              >
                <IconClose size={14} />
              </button>
            )}
          </div>

          <div className="chat-model-picker-list">
            {filteredLocal.length > 0 && (
              <div className="chat-model-picker-section">
                <div className="chat-model-picker-section-title">Downloaded</div>
                {filteredLocal.map((m) => (
                  <button
                    key={m.id}
                    type="button"
                    role="option"
                    aria-selected={m.id === selection}
                    className={`chat-model-picker-option${m.id === selection ? " active" : ""}`}
                    onClick={() => pickLocal(m.id)}
                  >
                    <span className="chat-model-picker-option-name">{modelLabel(m)}</span>
                    {m.source_label && (
                      <span className="chat-model-picker-option-meta">{m.source_label}</span>
                    )}
                  </button>
                ))}
              </div>
            )}

            {(hubModels.length > 0 || catalogLoading) && (
              <div className="chat-model-picker-section">
                <div className="chat-model-picker-section-title">Hugging Face Hub</div>
                {catalogLoading && hubModels.length === 0 && (
                  <div className="chat-model-picker-hint">Searching models…</div>
                )}
                {hubModels.map((m) => (
                  <button
                    key={m.repo_id}
                    type="button"
                    role="option"
                    className="chat-model-picker-option chat-model-picker-option-hub"
                    onClick={() => pickCatalog(m.repo_id)}
                  >
                    <span className="chat-model-picker-option-name">{m.name}</span>
                    <span className="chat-model-picker-option-meta">
                      {m.repo_id}
                      {m.download_bytes ? ` · ${formatBytes(m.download_bytes)} download` : m.params ? ` · ${m.params}` : ""}
                      {m.hardware_fit_label ? ` · ${m.hardware_fit_label}` : ""}
                    </span>
                  </button>
                ))}
              </div>
            )}

            {empty && <div className="chat-model-picker-hint">No models match your search.</div>}
          </div>
        </div>
      )}
    </div>
  );
}
