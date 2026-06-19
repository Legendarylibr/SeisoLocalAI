import { useCallback, useEffect, useMemo, useState } from "react";
import { api, CatalogModel, InferenceModelOption } from "@/lib/api";
import { inventoryMatchesRepo } from "@/lib/hubDownload";
import { formatBytes } from "@/lib/modelProgress";
import { modelMemoryBlocked, modelMemoryBlockReason } from "@/lib/chatModel";
import { HubComboboxSearch } from "@/components/HubComboboxSearch";
import { IconChevronDown } from "@/components/Icons";
import { useHubCombobox } from "@/hooks/useHubCombobox";

type ChatModelPickerProps = {
  models: InferenceModelOption[];
  selection: string;
  disabled?: boolean;
  switching?: boolean;
  headroomMb?: number;
  modelLabel: (m: InferenceModelOption) => string;
  onSelectLocal: (modelId: string) => void | Promise<void>;
  onSelectCatalog: (model: CatalogModel) => void | Promise<void>;
};

export function ChatModelPicker({
  models,
  selection,
  disabled,
  switching,
  headroomMb,
  modelLabel,
  onSelectLocal,
  onSelectCatalog,
}: ChatModelPickerProps) {
  const { open, setOpen, search, setSearch, rootRef, searchRef } = useHubCombobox();
  const [catalog, setCatalog] = useState<CatalogModel[]>([]);
  const [catalogLoading, setCatalogLoading] = useState(false);

  const selected = useMemo(() => models.find((m) => m.id === selection) ?? null, [models, selection]);

  const refreshCatalog = useCallback((q: string) => {
    setCatalogLoading(true);
    api
      .catalog(q, undefined, "chat", false)
      .then((r) => setCatalog(r.models))
      .catch(() => setCatalog([]))
      .finally(() => setCatalogLoading(false));
  }, []);

  useEffect(() => {
    if (!open) return;
    const t = setTimeout(() => refreshCatalog(search), search ? 180 : 0);
    return () => clearTimeout(t);
  }, [open, search, refreshCatalog]);

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
    () => catalog.filter((c) => !models.some((m) => inventoryMatchesRepo(m, c.repo_id)) && c.download_available !== false),
    [catalog, models],
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

  const pickCatalog = (model: CatalogModel) => {
    setOpen(false);
    void onSelectCatalog(model);
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
          <HubComboboxSearch
            searchRef={searchRef}
            value={search}
            placeholder="Search local & Hugging Face models…"
            onChange={setSearch}
            onEscape={() => setOpen(false)}
          />

          <div className="chat-model-picker-list">
            {filteredLocal.length > 0 && (
              <div className="chat-model-picker-section">
                <div className="chat-model-picker-section-title">Downloaded</div>
                {filteredLocal.map((m) => {
                  const blocked = modelMemoryBlocked(m, headroomMb);
                  return (
                  <button
                    key={m.id}
                    type="button"
                    role="option"
                    aria-selected={m.id === selection}
                    aria-disabled={blocked}
                    disabled={blocked}
                    className={`chat-model-picker-option${m.id === selection ? " active" : ""}${blocked ? " blocked" : ""}`}
                    title={blocked ? modelMemoryBlockReason(m) : undefined}
                    onClick={() => !blocked && pickLocal(m.id)}
                  >
                    <span className="chat-model-picker-option-name">{modelLabel(m)}</span>
                    {m.source_label && (
                      <span className="chat-model-picker-option-meta">
                        {blocked ? modelMemoryBlockReason(m) : m.source_label}
                      </span>
                    )}
                  </button>
                  );
                })}
              </div>
            )}

            {(hubModels.length > 0 || catalogLoading) && (
              <div className="chat-model-picker-section">
                <div className="chat-model-picker-section-title">Hugging Face Hub</div>
                {catalogLoading && hubModels.length === 0 && (
                  <div className="chat-model-picker-hint">Searching models…</div>
                )}
                {hubModels.map((m) => {
                  const blocked = modelMemoryBlocked(m, headroomMb);
                  return (
                  <button
                    key={m.repo_id}
                    type="button"
                    role="option"
                    aria-disabled={blocked}
                    disabled={blocked}
                    className={`chat-model-picker-option chat-model-picker-option-hub${blocked ? " blocked" : ""}`}
                    title={blocked ? modelMemoryBlockReason(m) : undefined}
                    onClick={() => !blocked && pickCatalog(m)}
                  >
                    <span className="chat-model-picker-option-name">{m.name}</span>
                    <span className="chat-model-picker-option-meta">
                      {blocked
                        ? modelMemoryBlockReason(m)
                        : <>
                          {m.repo_id}
                          {m.download_bytes
                            ? ` · ${m.download_bytes_estimated ? "~" : ""}${formatBytes(m.download_bytes)} download`
                            : m.params
                              ? ` · ${m.params}`
                              : ""}
                          {m.hardware_fit_label ? ` · ${m.hardware_fit_label}` : ""}
                          {m.download_mirror_verified === false && m.download_error ? " · mirror not pre-verified" : ""}
                        </>}
                    </span>
                  </button>
                  );
                })}
              </div>
            )}

            {empty && <div className="chat-model-picker-hint">No models match your search.</div>}
          </div>
        </div>
      )}
    </div>
  );
}
