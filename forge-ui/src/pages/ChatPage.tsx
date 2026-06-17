import { useEffect, useRef, useState } from "react";
import { api, ChatMessage, ChatThread, LocalModel } from "@/lib/api";

export function ChatPage() {
  const [threads, setThreads] = useState<ChatThread[]>([]);
  const [active, setActive] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [logs, setLogs] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [useTools, setUseTools] = useState(false);
  const [providerId, setProviderId] = useState<string>("");
  const [modelId, setModelId] = useState<string>("");
  const [models, setModels] = useState<LocalModel[]>([]);
  const [providers, setProviders] = useState<Array<{ id: string; name: string; provider_type: string }>>([]);
  const bottomRef = useRef<HTMLDivElement>(null);

  const [vramModel, setVramModel] = useState<string | null>(null);

  useEffect(() => {
    api.listThreads().then(setThreads).catch(console.error);
    api.listModels().then((m) => {
      setModels(m);
      if (m.length && !modelId) setModelId(m[0].id);
    }).catch(console.error);
    api.listProviders().then(setProviders).catch(() => {});
    api.vramStatus().then((s) => setVramModel(s.active_model)).catch(() => {});
  }, []);

  const switchModel = async (newModelId: string) => {
    if (newModelId === modelId) return;
    setModelId(newModelId);
    setProviderId("");
    try {
      await api.unloadVram();
      setVramModel(null);
    } catch {
      /* ignore */
    }
  };

  useEffect(() => {
    if (active) api.getMessages(active).then(setMessages).catch(console.error);
  }, [active]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, logs]);

  const newThread = async () => {
    const t = await api.createThread("New chat");
    setThreads((prev) => [t, ...prev]);
    setActive(t.id);
    setMessages([]);
  };

  const send = async () => {
    if (!input.trim() || streaming) return;
    const content = input.trim();
    setInput("");
    setStreaming(true);
    setLogs([]);
    setError(null);

    let threadId = active;
    if (!threadId) {
      const t = await api.createThread(content.slice(0, 40));
      threadId = t.id;
      setActive(threadId);
      setThreads((prev) => [t, ...prev]);
    }

    setMessages((prev) => [
      ...prev,
      { id: crypto.randomUUID(), role: "user", content, created_at: new Date().toISOString() },
    ]);

    const history = [
      ...messages.map((m) => ({ role: m.role, content: m.content })),
      { role: "user", content },
    ];

    try {
      const res = await fetch("/api/inference/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${localStorage.getItem("seiso_token") || ""}`,
        },
        credentials: "include",
        body: JSON.stringify({
          thread_id: threadId,
          messages: history,
          stream: true,
          tools: useTools,
          provider_id: providerId || null,
          model_id: providerId ? null : modelId || null,
        }),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || "Chat request failed");
      }

      const reader = res.body?.getReader();
      const decoder = new TextDecoder();
      let assistantText = "";
      let buffer = "";

      if (reader) {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const blocks = buffer.split("\n\n");
          buffer = blocks.pop() || "";
          for (const block of blocks) {
            let event = "message";
            let data = "";
            for (const line of block.split("\n")) {
              if (line.startsWith("event:")) event = line.slice(6).trim();
              if (line.startsWith("data:")) data = line.slice(5).trim();
            }
            if (event === "log") setLogs((l) => [...l, data]);
            if (event === "error") setError(data);
            if (event === "token") {
              assistantText += data;
              setMessages((prev) => {
                const copy = [...prev];
                const last = copy[copy.length - 1];
                if (last?.role === "assistant") {
                  copy[copy.length - 1] = { ...last, content: assistantText };
                } else {
                  copy.push({
                    id: crypto.randomUUID(),
                    role: "assistant",
                    content: assistantText,
                    created_at: new Date().toISOString(),
                  });
                }
                return copy;
              });
            }
            if (event === "message") {
              assistantText = data;
              setMessages((prev) => {
                const copy = [...prev];
                const last = copy[copy.length - 1];
                if (last?.role === "assistant") {
                  copy[copy.length - 1] = { ...last, content: assistantText };
                } else {
                  copy.push({
                    id: crypto.randomUUID(),
                    role: "assistant",
                    content: assistantText,
                    created_at: new Date().toISOString(),
                  });
                }
                return copy;
              });
            }
          }
        }
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Request failed");
    } finally {
      setStreaming(false);
    }
  };

  return (
    <div>
      <h1 className="page-title">Chat</h1>
      <p className="page-sub">Local inference with tools, providers, and SSE streaming.</p>

      <div className="card" style={{ marginBottom: "1rem", display: "flex", gap: "1rem", flexWrap: "wrap", alignItems: "center" }}>
        <label style={{ display: "flex", alignItems: "center", gap: "0.4rem", margin: 0 }}>
          <input type="checkbox" checked={useTools} onChange={(e) => setUseTools(e.target.checked)} />
          Enable tools (web search, code, artifacts)
        </label>
        <div style={{ minWidth: "200px" }}>
          <label>Local model</label>
          <select
            value={modelId}
            onChange={(e) => switchModel(e.target.value)}
            disabled={!!providerId}
            style={{ margin: 0 }}
          >
            {models.length === 0 && <option value="">No models — download from Hub</option>}
            {models.map((m) => (
              <option key={m.id} value={m.id}>{m.name} ({m.format})</option>
            ))}
          </select>
        </div>
        <div style={{ minWidth: "200px" }}>
          <label>Provider (optional)</label>
          <select
            value={providerId}
            onChange={(e) => setProviderId(e.target.value)}
            style={{ margin: 0 }}
          >
            <option value="">Local model</option>
            {providers.map((p) => (
              <option key={p.id} value={p.id}>{p.name} ({p.provider_type})</option>
            ))}
          </select>
        </div>
        {vramModel && (
          <span className="vram-indicator" title="Model loaded in VRAM">
            VRAM: {vramModel.split(":").pop()?.slice(0, 30)}…
          </span>
        )}
      </div>

      <div className="chat-layout">
        <div className="card thread-list">
          <button className="btn btn-primary" style={{ width: "100%", marginBottom: "0.75rem" }} onClick={newThread}>
            + New thread
          </button>
          {threads.map((t) => (
            <button
              key={t.id}
              className="btn"
              style={{
                width: "100%",
                marginBottom: "0.35rem",
                textAlign: "left",
                borderColor: active === t.id ? "var(--accent)" : undefined,
              }}
              onClick={() => setActive(t.id)}
            >
              {t.title}
            </button>
          ))}
        </div>

        <div className="card" style={{ display: "flex", flexDirection: "column" }}>
          <div className="messages">
            {messages.map((m) => (
              <div key={m.id} className={`msg msg-${m.role}`}>
                {m.content}
              </div>
            ))}
            <div ref={bottomRef} />
          </div>
          {logs.length > 0 && <div className="log-panel">{logs.join("\n")}</div>}
          {error && <p style={{ color: "var(--danger)", padding: "0.5rem 0" }}>{error}</p>}
          <div className="composer">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Message…"
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  send();
                }
              }}
            />
            <button className="btn btn-primary" onClick={send} disabled={streaming}>
              {streaming ? "…" : "Send"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
