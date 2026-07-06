import { useMemo } from "react";
import type { ModelVariantsResponse } from "@/lib/api/types";
import {
  ContextWindowSetting,
  formatContextLabel,
} from "@/lib/chatContext";
import type { ChatInferenceSettings } from "@/lib/chatInferenceSettings";
import { formatBytes } from "@/lib/modelProgress";
import { IconChevronDown } from "@/components/Icons";

type ChatInferencePanelProps = {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  settings: ChatInferenceSettings;
  onSettingsChange: (partial: Partial<ChatInferenceSettings>) => void;
  contextWindow: ContextWindowSetting;
  onContextWindowChange: (value: ContextWindowSetting) => void;
  contextWindowOptions: number[];
  variants: ModelVariantsResponse | null;
  variantsLoading?: boolean;
  downloadingQuant: string | null;
  disabled?: boolean;
  providerActive?: boolean;
  onSelectLocalVariant: (modelId: string) => void;
  onDownloadHubVariant: (repo: string, filename: string, quant: string) => void;
};

const MAX_TOKEN_OPTIONS = [512, 1024, 2048, 4096, 8192];
const SPEC_TOKEN_OPTIONS = [2, 4, 6, 8, 12, 16, 24, 32];

function formatMaxTokens(value: number): string {
  if (value >= 1024) return `${value / 1024}k`;
  return String(value);
}

export function ChatInferencePanel({
  open,
  onOpenChange,
  settings,
  onSettingsChange,
  contextWindow,
  onContextWindowChange,
  contextWindowOptions,
  variants,
  variantsLoading = false,
  downloadingQuant,
  disabled = false,
  providerActive = false,
  onSelectLocalVariant,
  onDownloadHubVariant,
}: ChatInferencePanelProps) {
  const contextOptions = useMemo<ContextWindowSetting[]>(
    () => ["auto", ...contextWindowOptions],
    [contextWindowOptions],
  );

  const quantRows = useMemo(() => {
    if (!variants) return [];
    const seen = new Set<string>();
    const rows: Array<{
      key: string;
      quant: string;
      selected: boolean;
      downloaded: boolean;
      localId?: string | null;
      repo?: string;
      filename?: string;
      sizeBytes?: number;
      blocked?: boolean;
      label?: string;
    }> = [];

    for (const row of variants.local_variants) {
      const key = `local:${row.id}`;
      seen.add(row.quant);
      rows.push({
        key,
        quant: row.quant,
        selected: Boolean(row.selected),
        downloaded: true,
        localId: row.id,
        sizeBytes: row.size_bytes,
        blocked: row.memory_load_blocked,
        label: row.hardware_fit_label ?? undefined,
      });
    }

    for (const row of variants.hub_variants) {
      if (seen.has(row.quant)) continue;
      rows.push({
        key: `hub:${row.gguf_file}`,
        quant: row.quant,
        selected: false,
        downloaded: false,
        repo: row.gguf_repo,
        filename: row.gguf_file,
        blocked: row.memory_load_blocked,
        label: row.hardware_fit_label ?? undefined,
      });
    }

    return rows.sort((a, b) => a.quant.localeCompare(b.quant));
  }, [variants]);

  const specAvailable =
    !providerActive && Boolean(variants?.supports_speculative && variants.draft_candidates.length > 0);
  const showQuants = !providerActive && quantRows.length > 1;

  const summaryParts = [
    formatContextLabel(contextWindow),
    formatMaxTokens(settings.maxTokens),
    `temp ${settings.temperature.toFixed(1)}`,
  ];
  if (settings.specEnabled && settings.draftModelId) {
    summaryParts.push("spec decode");
  }
  if (variants?.current_quant) {
    summaryParts.push(variants.current_quant);
  }

  return (
    <section className={`chat-inference-panel${open ? " chat-inference-panel-open" : ""}`}>
      <button
        type="button"
        className="chat-inference-toggle"
        onClick={() => onOpenChange(!open)}
        aria-expanded={open}
      >
        <span className="chat-inference-toggle-text">
          <span className="chat-inference-title">Inference settings</span>
          <span className="chat-inference-summary muted-text">{summaryParts.join(" · ")}</span>
        </span>
        <IconChevronDown size={16} className="chat-inference-chevron" />
      </button>

      {open && (
        <div className="chat-inference-body">
          <div className="chat-inference-grid">
            <label className="chat-inference-field">
              <span className="muted-text">Context window</span>
              <select
                value={String(contextWindow)}
                disabled={disabled}
                onChange={(e) => {
                  const raw = e.target.value;
                  onContextWindowChange(raw === "auto" ? "auto" : Number(raw));
                }}
                title="Context size — Auto sizes to prompt + reply; max depends on model and free VRAM"
              >
                {contextOptions.map((opt) => (
                  <option key={String(opt)} value={String(opt)}>
                    {formatContextLabel(opt)}
                  </option>
                ))}
              </select>
            </label>

            <label className="chat-inference-field">
              <span className="muted-text">Max reply</span>
              <select
                value={settings.maxTokens}
                disabled={disabled}
                onChange={(e) => onSettingsChange({ maxTokens: Number(e.target.value) })}
              >
                {MAX_TOKEN_OPTIONS.map((opt) => (
                  <option key={opt} value={opt}>
                    {formatMaxTokens(opt)}
                  </option>
                ))}
              </select>
            </label>

            <label className="chat-inference-field">
              <span className="muted-text">Temperature</span>
              <div className="chat-inference-range">
                <input
                  type="range"
                  min={0}
                  max={2}
                  step={0.1}
                  value={settings.temperature}
                  disabled={disabled}
                  onChange={(e) => onSettingsChange({ temperature: Number(e.target.value) })}
                />
                <span className="chat-inference-value">{settings.temperature.toFixed(1)}</span>
              </div>
            </label>

            <label className="chat-inference-field chat-inference-field-check">
              <span className="muted-text">Top-p</span>
              <div className="chat-inference-inline">
                <input
                  type="checkbox"
                  checked={settings.topPEnabled}
                  disabled={disabled || settings.temperature <= 0}
                  onChange={(e) => onSettingsChange({ topPEnabled: e.target.checked })}
                />
                <input
                  type="range"
                  min={0.05}
                  max={1}
                  step={0.05}
                  value={settings.topP}
                  disabled={disabled || !settings.topPEnabled || settings.temperature <= 0}
                  onChange={(e) => onSettingsChange({ topP: Number(e.target.value) })}
                />
                <span className="chat-inference-value">
                  {settings.topPEnabled ? settings.topP.toFixed(2) : "off"}
                </span>
              </div>
            </label>
          </div>

          {showQuants && (
            <div className="chat-inference-section">
              <div className="chat-inference-section-head">
                <span className="chat-inference-section-title">Quant variants</span>
                {variantsLoading && <span className="muted-text">Refreshing…</span>}
                {variants?.current_quant && (
                  <span className="chat-inference-current-quant muted-text">
                    Current: {variants.current_quant}
                  </span>
                )}
              </div>
              <div className="chat-inference-quants">
                {quantRows.map((row) => {
                  const busy = downloadingQuant === row.quant;
                  const title = [
                    row.quant,
                    row.sizeBytes ? formatBytes(row.sizeBytes) : null,
                    row.label,
                    row.blocked ? "May exceed available memory" : null,
                  ]
                    .filter(Boolean)
                    .join(" · ");

                  if (row.downloaded && row.localId) {
                    return (
                      <button
                        key={row.key}
                        type="button"
                        className={`chat-quant-chip${row.selected ? " chat-quant-chip-active" : ""}${row.blocked ? " chat-quant-chip-blocked" : ""}`}
                        disabled={disabled || busy || row.selected}
                        title={title}
                        onClick={() => onSelectLocalVariant(row.localId!)}
                      >
                        {row.quant}
                      </button>
                    );
                  }

                  return (
                    <button
                      key={row.key}
                      type="button"
                      className={`chat-quant-chip chat-quant-chip-download${row.blocked ? " chat-quant-chip-blocked" : ""}`}
                      disabled={disabled || busy || !row.repo || !row.filename}
                      title={title || `Download ${row.quant}`}
                      onClick={() => {
                        if (row.repo && row.filename) {
                          onDownloadHubVariant(row.repo, row.filename, row.quant);
                        }
                      }}
                    >
                      {busy ? "…" : `${row.quant} ↓`}
                    </button>
                  );
                })}
              </div>
            </div>
          )}

          {!providerActive && variants && !variants.supports_speculative && settings.specEnabled && (
            <p className="chat-inference-note muted-text">
              Speculative decoding needs a PyTorch-capable main model (safetensors), not GGUF-only.
            </p>
          )}

          {specAvailable && (
            <div className="chat-inference-section">
              <div className="chat-inference-section-head">
                <label className="chat-inference-spec-toggle">
                  <input
                    type="checkbox"
                    checked={settings.specEnabled}
                    disabled={disabled}
                    onChange={(e) => onSettingsChange({ specEnabled: e.target.checked })}
                  />
                  <span className="chat-inference-section-title">Speculative decoding</span>
                </label>
                <span className="muted-text">Uses PyTorch with a smaller draft model</span>
              </div>

              {settings.specEnabled && (
                <div className="chat-inference-grid">
                  <label className="chat-inference-field">
                    <span className="muted-text">Draft model</span>
                    <select
                      value={settings.draftModelId}
                      disabled={disabled}
                      onChange={(e) => onSettingsChange({ draftModelId: e.target.value })}
                    >
                      <option value="">Select draft…</option>
                      {variants!.draft_candidates.map((draft) => (
                        <option key={draft.id} value={draft.id}>
                          {draft.name || draft.id}
                          {draft.size_bytes ? ` · ${formatBytes(draft.size_bytes)}` : ""}
                        </option>
                      ))}
                    </select>
                  </label>

                  <label className="chat-inference-field">
                    <span className="muted-text">Draft tokens</span>
                    <select
                      value={settings.numSpeculativeTokens}
                      disabled={disabled}
                      onChange={(e) =>
                        onSettingsChange({ numSpeculativeTokens: Number(e.target.value) })
                      }
                    >
                      {SPEC_TOKEN_OPTIONS.map((opt) => (
                        <option key={opt} value={opt}>
                          {opt}
                        </option>
                      ))}
                    </select>
                  </label>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </section>
  );
}
