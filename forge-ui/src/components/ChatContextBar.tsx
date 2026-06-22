import type { ChatContextStatus } from "@/lib/api/types";
import {
  contextSummary,
  contextWindowFillClass,
} from "@/lib/chatContext";

type ChatContextBarProps = {
  status: ChatContextStatus | null;
  loading?: boolean;
};

export function ChatContextBar({
  status,
  loading = false,
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
    </section>
  );
}
