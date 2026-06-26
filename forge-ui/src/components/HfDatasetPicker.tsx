import { useCallback, useEffect, useState } from "react";
import { api, CatalogDataset } from "@/lib/api";
import { HubComboboxSearch } from "@/components/HubComboboxSearch";
import { IconChevronDown } from "@/components/Icons";
import { useHubCombobox } from "@/hooks/useHubCombobox";

const BUNDLED_SAMPLE = "./data/sample.jsonl";

type HfDatasetPickerProps = {
  value: string;
  onChange: (repoId: string) => void;
  disabled?: boolean;
};

function isLocalDatasetPath(value: string): boolean {
  const trimmed = value.trim();
  return (
    trimmed.startsWith("./") ||
    trimmed.startsWith("../") ||
    trimmed.startsWith("~/") ||
    trimmed.startsWith("/")
  );
}

export function HfDatasetPicker({ value, onChange, disabled }: HfDatasetPickerProps) {
  const { open, setOpen, search, setSearch, rootRef, searchRef } = useHubCombobox();
  const [results, setResults] = useState<CatalogDataset[]>([]);
  const [loading, setLoading] = useState(false);

  const refreshResults = useCallback((q: string) => {
    const trimmed = q.trim();
    if (!trimmed || isLocalDatasetPath(trimmed)) {
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
    if (open) setResults([]);
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

  const q = search.trim();
  const customIsLocal = isLocalDatasetPath(q);
  const empty = !loading && q && results.length === 0;
  const showCustomHint = q && (customIsLocal || empty);

  const triggerLabel = value || "Search datasets or enter a path…";

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
        <span className="chat-model-picker-label">{triggerLabel}</span>
        <span className="chat-model-picker-chevron" aria-hidden>
          <IconChevronDown size={14} />
        </span>
      </button>

      {open && (
        <div className="chat-model-picker-menu" role="listbox">
          <HubComboboxSearch
            searchRef={searchRef}
            value={search}
            placeholder="Search Hugging Face datasets or paste a local path…"
            onChange={setSearch}
            onEscape={() => setOpen(false)}
            onEnter={applyCustom}
          />

          <div className="chat-model-picker-list">
            <div className="chat-model-picker-section">
              <div className="chat-model-picker-section-title">Quick start</div>
              <button
                type="button"
                role="option"
                aria-selected={value === BUNDLED_SAMPLE}
                className={`chat-model-picker-option${value === BUNDLED_SAMPLE ? " active" : ""}`}
                onClick={() => pick(BUNDLED_SAMPLE)}
              >
                <span className="chat-model-picker-option-name">Bundled sample dataset</span>
                <span className="chat-model-picker-option-meta">
                  {BUNDLED_SAMPLE} · 4 chat rows for smoke tests
                </span>
              </button>
            </div>

            {!q && (
              <div className="chat-model-picker-hint">
                Type to search Hugging Face datasets, or press Enter to use a hub ID or local path.
              </div>
            )}
            {loading && q && !customIsLocal && results.length === 0 && (
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
            {showCustomHint && (
              <button
                type="button"
                role="option"
                className="chat-model-picker-option chat-model-picker-option-hub"
                onClick={applyCustom}
              >
                <span className="chat-model-picker-option-name">
                  {customIsLocal ? "Use local dataset path" : "Use exact dataset ID"}
                </span>
                <span className="chat-model-picker-option-meta">{q}</span>
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}