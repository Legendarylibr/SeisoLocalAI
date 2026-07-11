import { memo } from "react";
import type { ChatMessage } from "@/lib/api";
import { IconAssistant } from "@/components/Icons";

export const ChatBubble = memo(function ChatBubble({
  message,
  truncated,
}: {
  message: ChatMessage;
  /** Reply still hit max length after any auto-continues. */
  truncated?: boolean;
}) {
  return (
    <div className={`chat-bubble chat-bubble-${message.role}`}>
      <div className="chat-avatar">
        {message.role === "user" ? (
          <span className="chat-avatar-text">You</span>
        ) : (
          <IconAssistant size={14} />
        )}
      </div>
      <div className="chat-bubble-body">
        <div className="chat-bubble-content">{message.content}</div>
        {truncated && message.role === "assistant" && (
          <p className="chat-reply-truncated muted-text">
            Stopped at max reply length. Raise Max reply in Inference settings, or send “Continue”.
          </p>
        )}
      </div>
    </div>
  );
});
