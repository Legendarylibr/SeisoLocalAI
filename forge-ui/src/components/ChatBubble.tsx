import { memo } from "react";
import type { ChatMessage } from "@/lib/api";
import { IconAssistant } from "@/components/Icons";

export const ChatBubble = memo(function ChatBubble({ message }: { message: ChatMessage }) {
  return (
    <div className={`chat-bubble chat-bubble-${message.role}`}>
      <div className="chat-avatar">
        {message.role === "user" ? (
          <span className="chat-avatar-text">You</span>
        ) : (
          <IconAssistant size={14} />
        )}
      </div>
      <div className="chat-bubble-content">{message.content}</div>
    </div>
  );
});