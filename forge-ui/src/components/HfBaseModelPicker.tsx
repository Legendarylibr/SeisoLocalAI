import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, CatalogModel, TrainableModel } from "@/lib/api";
import { formatBytes } from "@/lib/modelProgress";
import { IconChevronDown, IconClose, IconSearch } from "@/components/Icons";

type HfBaseModelPickerProps = {
  value: string;
  localModels: TrainableModel[];
  disabled?: boolean;
  onChange: (value: string) => void;
};

export function HfBaseModelPicker({ value, localModels, disabled, onChange }: HfBaseModelPickerProps) {
  const [open, setOpen] = useState(false);
  const [search, setSearch] = useState("");
  const [results, setResults] = useState<CatalogModel[]>([]);
  const [loading, setLoading] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);
  const searchRef = useRef<HTMLInputElement>(null);

  const localByRepo = useMemo(() => {
    const map = new Map<string, TrainableModel>();
    for (const m of localModels) {
      if (m.repo_id) map.set(m.repo_id, m);
    }
    return map;
  }, [localModels]);

  const refreshResults = useCallback((q: string) => {
    setLoading(true);
    api
      .catalog(q.trim(), undefined, undefined, false)
      .then((r) => setResults(r.models))
      .catch(() => setResults([]))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    if (!open) return;
    const t = setTimeout(() => refreshResults(search), search ? 180 : 0);
    return () => clearTimeout(t);
  }, [open, search, refreshResults]);

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
      setResults([]);
      requestAnimationFrame(() => searchRef.current?.focus());
    }
  }, [open]);

  const pick = (repoId: string) => {
    setOpen(false);
    onChange(repoId);
  };

  const applyCustom = () => {
    const trimmed = search.trim();
    if (!trimmed) return;
    setOpen(false);
    onChange(trimmed);
  };

  const selectedLocal = useMemo(
    () => localModels.find((m) => m.repo_id === value || m.path === value),
    [localModels, value],
  );

  const selectedHub = useMemo(
    () => results.find((m) => m.repo_id === value) ?? localByRepo.has(value),
    // If the current value is a cached repo id, treat it as the local pick.
    [results, localByRepo, value],
  );

  const triggerLabel = selectedLocal
    ? selectedLocal.name
    : (selectedHub && typeof selectedHub !== "boolean")
      ? selectedHub.name
      : value || "Select or search a model…";

  const q = search.toLowerCase().trim();

  const hubModels = useMemo(() => {
    const localRepoIds = new Set(localModels.map((m) => m.repo_id).filter(Boolean) as string[]);
    if (!q) return results.filter((m) => !localRepoIds.has(m.repo_id));
    return results.filter((m) => !localRepoIds.has(m.repo_id));
  }, [results, localModels, q]);

  const filteredLocal = useMemo(() => {
    if (!q) return localModels;
    return localModels.filter(
      (m) =>
        m.name.toLowerCase().includes(q) ||
        (m.repo_id?.toLowerCase().includes(q) ?? false) ||
        m.path.toLowerCase().includes(q),
    );
  }, [localModels, q]);

  const emptyLocal = filteredLocal.length === 0;
  const emptyHub = !loading && hubModels.length === 0;
  const showCustomHint = q && emptyHub;

  return (
    <div className="chat-model-picker" ref={rootRef}>
      <button
        type="button"
        className="chat-model-picker-trigger"
        onClick={() => !disabled && setOpen((v) => !v)}
        disabled={disabled}
        aria-haspopup="listbox"
        aria-expanded={open}
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
                if (e.key === "Enter") {
                  e.preventDefault();
                  applyCustom();
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
            {!emptyLocal && (
              <div className="chat-model-picker-section">
                <div className="chat-model-picker-section-title">Cached locally</div>
                {filteredLocal.map((m) => {
                  const modelValue = m.repo_id || m.path;
                  return (
                    <button
                      key={m.id}
                      type="button"
                      role="option"
                      aria-selected={modelValue === value}
                      className={`chat-model-picker-option${modelValue === value ? " active" : ""}`}
                      onClick={() => pick(modelValue)}
                    >
                      <span className="chat-model-picker-option-name">{m.name}</span>
                      <span className="chat-model-picker-option-meta">
                        {m.repo_id || m.path}
                        {m.size_bytes ? ` · ${formatBytes(m.size_bytes)}` : ""}
                      </span>
                    </button>
                  );
                })}
              </div>
            )}

            {(hubModels.length > 0 || loading) && (
              <div className="chat-model-picker-section">
                <div className="chat-model-picker-section-title">Hugging Face Hub</div>
                {loading && hubModels.length === 0 && (
                  <div className="chat-model-picker-hint">Searching models…</div>
                )}
                {hubModels.map((m) => (
                  <button
                    key={m.repo_id}
                    type="button"
                    role="option"
                    aria-selected={m.repo_id === value}
                    className={`chat-model-picker-option chat-model-picker-option-hub${m.repo_id === value ? " active" : ""}`}
                    onClick={() => pick(m.repo_id)}
                  >
                    <span className="chat-model-picker-option-name">{m.name}</span>
                    <span className="chat-model-picker-option-meta">
                      {m.repo_id}
                      {m.download_bytes
                        ? ` · ${m.download_bytes_estimated ? "~" : ""}${formatBytes(m.download_bytes)} download`
                        : m.params
                          ? ` · ${m.params}`
                          : ""}
                      {m.hardware_fit_label ? ` · ${m.hardware_fit_label}` : ""}
                    </span>
                  </button>
                ))}
              </div>
            )}

            {emptyLocal && emptyHub && !showCustomHint && (
              <div className="chat-model-picker-hint">No models match your search.</div>
            )}

            {showCustomHint && (
              <button
                type="button"
                role="option"
                className="chat-model-picker-option chat-model-picker-option-hub"
                onClick={applyCustom}
              >
                <span className="chat-model-picker-option-name">Use exact repo ID</span>
                <span className="chat-model-picker-option-meta">{search.trim()}</span>
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
