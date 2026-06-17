import { useEffect, useMemo, useRef, useState } from "react";
import { api, ChatMessage, ChatThread, InferenceModelOption, SecurityPosture, streamChat } from "@/lib/api";

const BACKEND_LABELS: Record<string, string> = {
  auto: "Auto",
  llamacpp: "llama.cpp",
  ollama: "Ollama",
  mlx: "MLX",
  torch: "PyTorch",
};

export function ChatPage() {
  const [threads, setThreads] = useState<ChatThread[]>([]);
  const [active, setActive] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [logs, setLogs] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [useTools, setUseTools] = useState(false);
  const [allowCodeExec, setAllowCodeExec] = useState(false);
  const [mcpIds, setMcpIds] = useState<string[]>([]);
  const [providerId, setProviderId] = useState<string>("");
  const [selection, setSelection] = useState<string>("");
  const [inferenceBackend, setInferenceBackend] = useState<string>("auto");
  const [models, setModels] = useState<InferenceModelOption[]>([]);
  const [providers, setProviders] = useState<Array<{ id: string; name: string; provider_type: string }>>([]);
  const [mcpServers, setMcpServers] = useState<Array<{ id: string; name: string }>>([]);
  const [security, setSecurity] = useState<SecurityPosture | null>(null);
  const [toolsExpanded, setToolsExpanded] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  const [vramModel, setVramModel] = useState<string | null>(null);

  const selected = useMemo(
    () => models.find((m) => m.id === selection) ?? null,
    [models, selection],
  );

  const backendOptions = useMemo(() => {
    if (!selected || providerId) return ["auto"];
    const opts = new Set<string>(["auto", ...selected.backends]);
    return Array.from(opts);
  }, [selected, providerId]);

  const toolsAvailable = security?.allow_tools ?? false;
  const codeExecAvailable = security?.allow_code_exec ?? false;

  useEffect(() => {
    api.listThreads().then(setThreads).catch(console.error);
    api.listInferenceModels().then((r) => {
      setModels(r.models);
      if (r.models.length && !selection) {
        setSelection(r.models[0].id);
        setInferenceBackend("auto");
      }
    }).catch(console.error);
    api.listProviders().then(setProviders).catch(() => {});
    api.listMcpServers().then((s) => setMcpServers(s.map(({ id, name }) => ({ id, name })))).catch(() => {});
    api.settings().then((s) => setSecurity(s.security)).catch(() => {});
    api.vramStatus().then((s) => setVramModel(s.active_model)).catch(() => {});
  }, []);

  const switchModel = async (newSelection: string) => {
    if (newSelection === selection) return;
    setSelection(newSelection);
    setInferenceBackend("auto");
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

  const toggleMcp = (id: string) => {
    setMcpIds((prev) => (prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]));
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

    const isOllamaOnly = selected?.kind === "ollama";
    let assistantText = "";

    try {
      await streamChat(
        {
          thread_id: threadId,
          messages: history,
          stream: true,
          tools: useTools && toolsAvailable,
          allow_code_exec: allowCodeExec && codeExecAvailable,
          mcp_server_ids: useTools && toolsAvailable ? mcpIds : [],
          provider_id: providerId || null,
          model_id: providerId || isOllamaOnly ? null : selection || null,
          ollama_model: isOllamaOnly ? selected?.ollama_model : null,
          inference_backend: providerId ? "auto" : inferenceBackend,
        },
        {
          onEvent: (event, data) => {
            if (event === "log") setLogs((l) => [...l, data]);
            if (event === "error") setError(data);
            if (event === "token" || event === "message") {
              if (event === "message") assistantText = data;
              else assistantText += data;
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
          },
        },
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : "Request failed");
    } finally {
      setStreaming(false);
    }
  };

  const modelLabel = (m: InferenceModelOption) => {
    const engine = m.backend_labels[m.default_backend] || m.default_backend;
    const fmt = m.format ? ` · ${m.format}` : "";
    return `${m.name} (${m.source_label}${fmt} · ${engine})`;
  };

  return (
    <div>
      <h1 className="page-title">Chat</h1>
      <p className="page-sub">
        Local and provider models with optional agent tools — gated by your server security settings.
      </p>

      <div className="card chat-controls">
        <div style={{ minWidth: "260px", flex: 1 }}>
          <label>Model</label>
          <select
            value={selection}
            onChange={(e) => switchModel(e.target.value)}
            disabled={!!providerId}
            style={{ margin: 0 }}
          >
            {models.length === 0 && <option value="">No models — download from Hub or run export</option>}
            {models.map((m) => (
              <option key={m.id} value={m.id}>{modelLabel(m)}</option>
            ))}
          </select>
        </div>
        {selected && backendOptions.length > 1 && !providerId && (
          <div style={{ minWidth: "160px" }}>
            <label>Engine</label>
            <select
              value={inferenceBackend}
              onChange={(e) => setInferenceBackend(e.target.value)}
              style={{ margin: 0 }}
            >
              {backendOptions.map((b) => (
                <option key={b} value={b}>{BACKEND_LABELS[b] || selected.backend_labels[b] || b}</option>
              ))}
            </select>
          </div>
        )}
        <div style={{ minWidth: "200px" }}>
          <label>Provider (optional)</label>
          <select
            value={providerId}
            onChange={(e) => setProviderId(e.target.value)}
            style={{ margin: 0 }}
          >
            <option value="">Local engines</option>
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

      <div className="card tools-panel">
        <button
          type="button"
          className="tools-panel-toggle"
          onClick={() => setToolsExpanded((v) => !v)}
          aria-expanded={toolsExpanded}
        >
          <span className="tools-panel-icon">{toolsExpanded ? "▾" : "▸"}</span>
          <span>Agent tools</span>
          {!toolsAvailable && <span className="badge badge-dim">Server disabled</span>}
          {useTools && toolsAvailable && <span className="badge">Active</span>}
        </button>

        {toolsExpanded && (
          <div className="tools-panel-body">
            {!toolsAvailable ? (
              <p className="muted-text">
                Tools are off on this server. Set <code>SEISO_ALLOW_TOOLS=true</code> in your environment and restart to enable web search, artifacts, and MCP.
              </p>
            ) : (
              <>
                <label className="tool-check">
                  <input
                    type="checkbox"
                    checked={useTools}
                    onChange={(e) => {
                      setUseTools(e.target.checked);
                      if (!e.target.checked) setAllowCodeExec(false);
                    }}
                  />
                  <span>
                    <strong>Enable tools</strong>
                    <span className="muted-text"> — web search, artifact writes, MCP integrations</span>
                  </span>
                </label>

                {useTools && codeExecAvailable && (
                  <label className="tool-check tool-check-warn">
                    <input
                      type="checkbox"
                      checked={allowCodeExec}
                      onChange={(e) => setAllowCodeExec(e.target.checked)}
                    />
                    <span>
                      <strong>Allow code execution</strong>
                      <span className="muted-text"> — sandboxed Python (requires explicit opt-in)</span>
                    </span>
                  </label>
                )}

                {useTools && !codeExecAvailable && (
                  <p className="muted-text tool-hint">
                    Code execution is disabled. Set <code>SEISO_ALLOW_CODE_EXEC=true</code> to enable.
                  </p>
                )}

                {useTools && mcpServers.length > 0 && (
                  <div className="mcp-picker">
                    <div className="muted-text" style={{ marginBottom: "0.35rem", fontSize: "0.85rem" }}>MCP servers</div>
                    {mcpServers.map((s) => (
                      <label key={s.id} className="tool-check">
                        <input
                          type="checkbox"
                          checked={mcpIds.includes(s.id)}
                          onChange={() => toggleMcp(s.id)}
                        />
                        <span>{s.name}</span>
                      </label>
                    ))}
                  </div>
                )}

                {useTools && mcpServers.length === 0 && (
                  <p className="muted-text tool-hint">
                    No MCP servers configured — add them in Integrations (sidebar).
                  </p>
                )}
              </>
            )}
          </div>
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
          {error && <p className="chat-error">{error}</p>}
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
