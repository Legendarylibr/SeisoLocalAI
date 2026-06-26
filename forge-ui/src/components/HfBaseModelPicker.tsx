import { useCallback, useEffect, useMemo, useState } from "react";
import { api, CatalogModel, TrainableModel } from "@/lib/api";
import { formatBytes } from "@/lib/modelProgress";
import { GGUF_TRAIN_ERROR, isGgufOnlyRepoId } from "@/lib/trainRepo";
import { HubComboboxSearch } from "@/components/HubComboboxSearch";
import { IconChevronDown } from "@/components/Icons";
import { useHubCombobox } from "@/hooks/useHubCombobox";

type HfBaseModelPickerProps = {
  value: string;
  localModels: TrainableModel[];
  disabled?: boolean;
  /** chat = GGUF catalog; train = safetensors checkpoints for LoRA/SFT */
  mode?: "chat" | "train";
  onChange: (value: string) => void;
};

export function HfBaseModelPicker({
  value,
  localModels,
  disabled,
  mode = "chat",
  onChange,
}: HfBaseModelPickerProps) {
  const trainableOnly = mode === "train";
  const { open, setOpen, search, setSearch, rootRef, searchRef } = useHubCombobox();
  const [results, setResults] = useState<CatalogModel[]>([]);
  const [resolvedValue, setResolvedValue] = useState<CatalogModel | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!value) {
      setResolvedValue(null);
      return;
    }
    if (localModels.some((m) => m.repo_id === value || m.path === value)) {
      setResolvedValue(null);
      return;
    }
    let cancelled = false;
    api
      .catalog(value, undefined, undefined, false, null, 50, trainableOnly ? "train" : "chat")
      .then((r) => {
        if (cancelled) return;
        const hit = r.models.find((m) => m.repo_id === value) ?? null;
        setResolvedValue(hit);
      })
      .catch(() => {
        if (!cancelled) setResolvedValue(null);
      });
    return () => {
      cancelled = true;
    };
  }, [value, localModels, trainableOnly]);

  const refreshResults = useCallback(
    (q: string) => {
      setLoading(true);
      api
        .catalog(q.trim(), undefined, undefined, false, null, 50, trainableOnly ? "train" : "chat")
        .then((r) => setResults(r.models))
        .catch(() => setResults([]))
        .finally(() => setLoading(false));
    },
    [trainableOnly],
  );

  useEffect(() => {
    if (!open) return;
    const t = setTimeout(() => refreshResults(search), search ? 180 : 0);
    return () => clearTimeout(t);
  }, [open, search, refreshResults]);

  useEffect(() => {
    if (open) setResults([]);
  }, [open]);

  const pick = (repoId: string) => {
    if (trainableOnly && isGgufOnlyRepoId(repoId)) return;
    setOpen(false);
    onChange(repoId);
  };

  const applyCustom = () => {
    const trimmed = search.trim();
    if (!trimmed) return;
    if (trainableOnly && isGgufOnlyRepoId(trimmed)) return;
    setOpen(false);
    onChange(trimmed);
  };

  const selectedLocal = useMemo(
    () => localModels.find((m) => m.repo_id === value || m.path === value),
    [localModels, value],
  );

  const selectedHub = useMemo(
    () => results.find((m) => m.repo_id === value) ?? resolvedValue,
    [results, value, resolvedValue],
  );

  const triggerLabel = selectedLocal
    ? selectedLocal.name
    : selectedHub
      ? selectedHub.name
      : value || (trainableOnly ? "Select a base model…" : "Select or search a model…");

  const q = search.toLowerCase().trim();
  const customIsGguf = trainableOnly && isGgufOnlyRepoId(search.trim());

  const hubModels = useMemo(() => {
    const localRepoIds = new Set(localModels.map((m) => m.repo_id).filter(Boolean) as string[]);
    return results.filter((m) => {
      if (localRepoIds.has(m.repo_id)) return false;
      if (trainableOnly && isGgufOnlyRepoId(m.repo_id, m.tags)) return false;
      return true;
    });
  }, [results, localModels, trainableOnly]);

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
  const showCustomHint = q && emptyHub && !customIsGguf;

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
          <HubComboboxSearch
            searchRef={searchRef}
            value={search}
            placeholder={
              trainableOnly
                ? "Search Hugging Face checkpoints…"
                : "Search local & Hugging Face models…"
            }
            onChange={setSearch}
            onEscape={() => setOpen(false)}
            onEnter={applyCustom}
          />

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

            {customIsGguf && (
              <div className="chat-model-picker-hint chat-model-picker-hint-warn">
                {GGUF_TRAIN_ERROR}
              </div>
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