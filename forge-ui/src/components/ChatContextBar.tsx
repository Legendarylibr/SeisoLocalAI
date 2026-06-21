import type { ChatContextStatus } from "@/lib/api/types";
import {
  ContextWindowSetting,
  contextSummary,
  contextWindowFillClass,
  formatContextLabel,
} from "@/lib/chatContext";

type ChatContextBarProps = {
  status: ChatContextStatus | null;
  contextWindow: ContextWindowSetting;
  loading?: boolean;
  disabled?: boolean;
  onContextWindowChange: (value: ContextWindowSetting) => void;
};

const CONTEXT_OPTIONS: ContextWindowSetting[] = ["auto", 2048, 4096, 8192];

export function ChatContextBar({
  status,
  contextWindow,
  loading = false,
  disabled = false,
  onContextWindowChange,
}: ChatContextBarProps) {
  const fillRatio = status?.fill_ratio ?? 0;
  const fillPct = Math.min(100, Math.max(0, fillRatio * 100));
  const fillClass = contextWindowFillClass(fillRatio);

  return (
    <section className="chat-context-bar" aria-label="Context window">
      <div className="chat-context-main">
        <div className="chat-context-track" aria-hidden>
          <div
            className={`chat-context-fill ${fillClass}`}
            style={{ width: `${fillPct}%` }}
          />
        </div>
        <div className="chat-context-meta">
          <span className="chat-context-label">Context</span>
          <span className="chat-context-stats">
            {loading && !status ? "Measuring…" : status ? contextSummary(status) : "No messages yet"}
          </span>
          {status?.history_trimmed && (
            <span className="chat-context-badge" title="Older turns are trimmed before inference">
              trimmed
            </span>
          )}
        </div>
      </div>
      <label className="chat-context-window">
        <span className="muted-text">Window</span>
        <select
          value={String(contextWindow)}
          disabled={disabled}
          onChange={(e) => {
            const raw = e.target.value;
            onContextWindowChange(
              raw === "auto" ? "auto" : (Number(raw) as 2048 | 4096 | 8192),
            );
          }}
          title="llama.cpp context size — Auto sizes to prompt + reply"
        >
          {CONTEXT_OPTIONS.map((opt) => (
            <option key={String(opt)} value={String(opt)}>
              {formatContextLabel(opt)}
            </option>
          ))}
        </select>
      </label>
    </section>
  );
}
