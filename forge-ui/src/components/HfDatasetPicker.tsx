import { useCallback, useEffect, useRef, useState } from "react";
import { api, CatalogDataset } from "@/lib/api";
import { IconChevronDown, IconClose, IconSearch } from "@/components/Icons";

type HfDatasetPickerProps = {
  value: string;
  onChange: (repoId: string) => void;
  disabled?: boolean;
};

export function HfDatasetPicker({ value, onChange, disabled }: HfDatasetPickerProps) {
  const [open, setOpen] = useState(false);
  const [search, setSearch] = useState("");
  const [results, setResults] = useState<CatalogDataset[]>([]);
  const [loading, setLoading] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);
  const searchRef = useRef<HTMLInputElement>(null);

  const refreshResults = useCallback((q: string) => {
    const trimmed = q.trim();
    if (!trimmed) {
      setResults([]);
      setLoading(false);
      return;
    }
    setLoading(true);
    api
      .searchDatasets(trimmed)
      .then((r) => setResults(r.datasets))
      .catch(() => setResults([]))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    if (!open) return;
    const t = setTimeout(() => refreshResults(search), 180);
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

  const empty = !loading && search.trim() && results.length === 0;

  return (
    <div className="chat-model-picker hub-search-picker" ref={rootRef}>
      <button
        type="button"
        className="chat-model-picker-trigger"
        onClick={() => !disabled && setOpen((v) => !v)}
        disabled={disabled}
        aria-haspopup="listbox"
        aria-expanded={open}
      >
        <span className="chat-model-picker-label">{value || "Search datasets…"}</span>
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
              placeholder="Search Hugging Face datasets…"
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
            {!search.trim() && (
              <div className="chat-model-picker-hint">Type to search Hugging Face datasets.</div>
            )}
            {loading && search.trim() && results.length === 0 && (
              <div className="chat-model-picker-hint">Searching datasets…</div>
            )}
            {results.map((d) => (
              <button
                key={d.repo_id}
                type="button"
                role="option"
                aria-selected={d.repo_id === value}
                className={`chat-model-picker-option chat-model-picker-option-hub${d.repo_id === value ? " active" : ""}`}
                onClick={() => pick(d.repo_id)}
              >
                <span className="chat-model-picker-option-name">{d.name}</span>
                <span className="chat-model-picker-option-meta">
                  {d.repo_id}
                  {d.downloads != null ? ` · ${d.downloads.toLocaleString()} downloads` : ""}
                  {d.tags.length > 0 ? ` · ${d.tags.join(", ")}` : ""}
                </span>
              </button>
            ))}
            {empty && <div className="chat-model-picker-hint">No datasets match your search.</div>}
          </div>
        </div>
      )}
    </div>
  );
}
