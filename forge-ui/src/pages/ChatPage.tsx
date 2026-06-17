import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { api, ChatMessage, ChatThread, HardwareProfile, InferenceModelOption, SecurityPosture, streamChat } from "@/lib/api";
import { bootstrapChatModels, hasChatNavTarget, initializeChatSession, isChatModelReady, needsHubDownload, preloadWithProgress, resolveInferenceBackend } from "@/lib/chatModel";
import { ModelProgressState, initialDownloadProgress, initialLoadProgress } from "@/lib/modelProgress";
import { ChatModelPicker } from "@/components/ChatModelPicker";
import { HardwareFitBadge } from "@/components/HardwareFitBadge";
import { ModelLoadProgress } from "@/components/ModelLoadProgress";
import {
  IconAssistant,
  IconChevronLeft,
  IconChevronRight,
  IconClose,
  IconLock,
  IconPlus,
  IconRefresh,
  IconSend,
} from "@/components/Icons";

const BACKEND_LABELS: Record<string, string> = {
  llamacpp: "llama.cpp",
  ollama: "Ollama",
  mlx: "MLX",
  torch: "PyTorch",
};

type OpenTab = { threadId: string; title: string };

export function ChatPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const pendingModel = searchParams.get("model");
  const pendingRepo = searchParams.get("repo");
  const pendingDownloadBytes = (() => {
    const raw = searchParams.get("bytes");
    if (!raw) return undefined;
    const n = Number(raw);
    return Number.isFinite(n) && n > 0 ? n : undefined;
  })();
  const navTarget = useMemo(
    () => ({ modelId: pendingModel, repo: pendingRepo, downloadBytes: pendingDownloadBytes }),
    [pendingModel, pendingRepo, pendingDownloadBytes],
  );
  const hasNavTarget = hasChatNavTarget(navTarget);
  const [threads, setThreads] = useState<ChatThread[]>([]);
  const [openTabs, setOpenTabs] = useState<OpenTab[]>([]);
  const [active, setActive] = useState<string | null>(null);
  const [messagesByThread, setMessagesByThread] = useState<Record<string, ChatMessage[]>>({});
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [threadSearch, setThreadSearch] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [useTools, setUseTools] = useState(false);
  const [allowCodeExec, setAllowCodeExec] = useState(false);
  const [providerId, setProviderId] = useState("");
  const [selection, setSelection] = useState("");
  const [inferenceBackend, setInferenceBackend] = useState("llamacpp");
  const [models, setModels] = useState<InferenceModelOption[]>([]);
  const [providers, setProviders] = useState<Array<{ id: string; name: string; provider_type: string }>>([]);
  const [security, setSecurity] = useState<SecurityPosture | null>(null);
  const [hwProfile, setHwProfile] = useState<HardwareProfile | null>(null);
  const [switchingModel, setSwitchingModel] = useState(false);
  const [loadProgress, setLoadProgress] = useState<ModelProgressState | null>(null);
  const [loadedModelId, setLoadedModelId] = useState<string | null>(null);
  const [loadedBackend, setLoadedBackend] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const streamAbortRef = useRef<(() => void) | null>(null);
  const streamTextRef = useRef("");
  const streamFlushRef = useRef<number | null>(null);
  const streamThreadRef = useRef<string | null>(null);
  const userPickedBackendRef = useRef(false);
  const bootstrapGenRef = useRef(0);
  const sessionInitRef = useRef(false);

  const messages = active ? messagesByThread[active] ?? [] : [];

  const selected = useMemo(() => models.find((m) => m.id === selection) ?? null, [models, selection]);
  const pendingModelLabel = useMemo(() => {
    if (!pendingRepo) return null;
    return pendingRepo.split("/").pop() || pendingRepo;
  }, [pendingRepo]);
  const backendOptions = useMemo(() => {
    if (!selected || providerId) return [];
    return selected.backends ?? [];
  }, [selected, providerId]);
  const effectiveBackend = useMemo(
    () => resolveInferenceBackend(selected, hwProfile, inferenceBackend),
    [selected, hwProfile, inferenceBackend],
  );
  const modelReady = useMemo(
    () => isChatModelReady(selection, effectiveBackend, loadedModelId, loadedBackend),
    [selection, effectiveBackend, loadedModelId, loadedBackend],
  );
  const showModelStatus = !providerId && !!(loadProgress || switchingModel || (hasNavTarget && !modelReady) || (selected && !modelReady));
  const waitingForModel = switchingModel || !!loadProgress;
  const effectiveLoadProgress =
    loadProgress ??
    (switchingModel && models.length > 0 && hasNavTarget && pendingRepo && needsHubDownload(models, navTarget)
      ? initialDownloadProgress(pendingRepo, pendingDownloadBytes)
      : null);
  const selectedFit = selected?.hardware_fit;

  const filteredThreads = useMemo(() => {
    const q = threadSearch.toLowerCase();
    if (!q) return threads;
    return threads.filter((t) => t.title.toLowerCase().includes(q));
  }, [threads, threadSearch]);

  const refreshModels = useCallback(async () => {
    const r = await api.listInferenceModels();
    setModels(r.models);
    if (selection) {
      const still = r.models.find((m) => m.id === selection);
      if (still) return r.models;
    }
    const pick = r.models.find((m) => m.hardware_fit === "ideal" || m.hardware_fit === "good")?.id || r.models[0]?.id;
    if (pick) {
      const model = r.models.find((m) => m.id === pick);
      setSelection(pick);
      setInferenceBackend(resolveInferenceBackend(model ?? null, hwProfile));
    }
    return r.models;
  }, [selection, hwProfile]);

  const activateModel = useCallback(
    async (modelId: string, list: InferenceModelOption[], backendOverride?: string) => {
      const next = list.find((m) => m.id === modelId);
      if (!next) {
        throw new Error("Model not found in inventory after download");
      }
      setSelection(modelId);
      const backend =
        backendOverride ??
        resolveInferenceBackend(next, hwProfile, userPickedBackendRef.current ? inferenceBackend : undefined);
      setInferenceBackend(backend);
      if (providerId) return backend;
      setLoadProgress(initialLoadProgress(next.name, next.size_bytes));
      const loaded = await preloadWithProgress(modelId, backend, setLoadProgress);
      setLoadedModelId(modelId);
      setLoadedBackend(loaded);
      return loaded;
    },
    [providerId, hwProfile, inferenceBackend],
  );

  const handleModelChange = async (modelId: string) => {
    const next = models.find((m) => m.id === modelId);
    const targetBackend = resolveInferenceBackend(next ?? null, hwProfile, inferenceBackend);
    if (
      modelId === selection &&
      isChatModelReady(modelId, targetBackend, loadedModelId, loadedBackend)
    ) {
      return;
    }
    setSwitchingModel(true);
    setError(null);
    setLoadProgress(null);
    streamAbortRef.current?.();
    streamAbortRef.current = null;
    if (streaming) setStreaming(false);
    if (!providerId) {
      try {
        await api.cancelInference();
        setLoadedModelId(null);
        setLoadedBackend(null);
      } catch {
        /* best-effort VRAM release */
      }
    }
    try {
      await activateModel(modelId, models);
      setSearchParams({ model: modelId }, { replace: true });
    } catch (e) {
      setLoadedModelId(null);
      setLoadedBackend(null);
      setError(e instanceof Error ? e.message : "Failed to load model into inference engine");
    } finally {
      setSwitchingModel(false);
      setLoadProgress(null);
    }
  };

  const handleCatalogSelect = (repoId: string, downloadBytes?: number) => {
    streamAbortRef.current?.();
    streamAbortRef.current = null;
    if (streaming) setStreaming(false);
    setError(null);
    const params = new URLSearchParams({ repo: repoId });
    if (downloadBytes && downloadBytes > 0) {
      params.set("bytes", String(downloadBytes));
    }
    setSearchParams(params, { replace: true });
  };

  const handleProviderChange = async (nextProvider: string) => {
    if (nextProvider === providerId) return;
    streamAbortRef.current?.();
    streamAbortRef.current = null;
    if (streaming) setStreaming(false);
    if (!nextProvider && providerId) {
      try {
        await api.cancelInference();
      } catch {
        /* ignore */
      }
    }
    setProviderId(nextProvider);
    if (!nextProvider && selection) {
      const next = models.find((m) => m.id === selection);
      if (next) {
        try {
          setLoadProgress(initialLoadProgress(next.name, next.size_bytes));
          const loaded = await preloadWithProgress(
            selection,
            resolveInferenceBackend(next, hwProfile, inferenceBackend),
            setLoadProgress,
          );
          setLoadedModelId(selection);
          setLoadedBackend(loaded);
        } catch {
          setLoadedModelId(null);
          setLoadedBackend(null);
          /* best-effort */
        } finally {
          setLoadProgress(null);
        }
      }
    }
  };

  const toolsAvailable = security?.allow_tools ?? false;
  const codeExecAvailable = security?.allow_code_exec ?? false;

  const loadMessages = useCallback(async (threadId: string) => {
    const msgs = await api.getMessages(threadId);
    setMessagesByThread((prev) => ({ ...prev, [threadId]: msgs }));
  }, []);

  useEffect(() => {
    api.listThreads().then(setThreads).catch(console.error);
    api.listProviders().then(setProviders).catch(() => {});
    api.settings().then((s) => setSecurity(s.security)).catch(() => {});
  }, []);

  useEffect(() => {
    if (!pendingRepo && sessionInitRef.current && (!pendingModel || pendingModel === selection)) {
      return;
    }

    const bootstrapGen = ++bootstrapGenRef.current;
    let cancelled = false;

    const applyResult = (result: {
      models: InferenceModelOption[];
      selectedId: string;
      backend: string;
    }) => {
      setModels(result.models);
      if (!result.selectedId) return;
      setSelection(result.selectedId);
      setInferenceBackend(result.backend);
      if (!providerId) {
        setLoadedModelId(result.selectedId);
        setLoadedBackend(result.backend);
      }
    };

    const bootstrap = async () => {
      setSwitchingModel(true);
      setError(null);
      setLoadedModelId(null);
      setLoadedBackend(null);
      try {
        const [hw, initialModelsResp] = await Promise.all([
          api.hardware().catch(() => null),
          api.listInferenceModels(),
        ]);
        if (cancelled) return;
        if (hw) setHwProfile(hw);

        const initialModels = initialModelsResp.models;
        const commonOptions = {
          preload: !providerId,
          providerActive: !!providerId,
          onProgress: setLoadProgress,
          initialModels,
          hwProfile: hw,
        };

        if (pendingRepo) {
          if (needsHubDownload(initialModels, navTarget)) {
            setLoadProgress(initialDownloadProgress(pendingRepo, pendingDownloadBytes));
          } else {
            setLoadProgress(null);
          }
          const result = await bootstrapChatModels(navTarget, commonOptions);
          if (cancelled) return;
          applyResult(result);
          sessionInitRef.current = true;
          if (result.selectedId) {
            setSearchParams({ model: result.selectedId }, { replace: true });
          } else {
            setSearchParams({}, { replace: true });
          }
        } else if (pendingModel) {
          setLoadProgress(null);
          const result = await bootstrapChatModels({ modelId: pendingModel }, commonOptions);
          if (cancelled) return;
          applyResult(result);
          sessionInitRef.current = true;
        } else {
          if (sessionInitRef.current) return;
          setLoadProgress(null);
          const result = await initializeChatSession(commonOptions);
          if (cancelled) return;
          applyResult(result);
          sessionInitRef.current = true;
        }
      } catch (e) {
        if (cancelled) return;
        if (e instanceof DOMException && e.name === "AbortError") return;
        setLoadedModelId(null);
        setLoadedBackend(null);
        setError(e instanceof Error ? e.message : "Failed to load models");
        if (pendingRepo) {
          setSearchParams({}, { replace: true });
        }
      } finally {
        if (bootstrapGen === bootstrapGenRef.current) {
          setSwitchingModel(false);
          setLoadProgress(null);
        }
      }
    };

    void bootstrap();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pendingModel, pendingRepo, pendingDownloadBytes]);

  useEffect(() => () => {
    sessionInitRef.current = false;
  }, []);

  useEffect(() => {
    if (active && !messagesByThread[active]) loadMessages(active).catch(console.error);
  }, [active, messagesByThread, loadMessages]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: streaming ? "auto" : "smooth" });
  }, [messages, active, streaming]);

  const openThread = (t: ChatThread) => {
    setActive(t.id);
    if (!openTabs.find((tab) => tab.threadId === t.id)) {
      setOpenTabs((prev) => [...prev, { threadId: t.id, title: t.title }]);
    }
  };

  const closeTab = (threadId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setOpenTabs((prev) => {
      const remaining = prev.filter((t) => t.threadId !== threadId);
      if (active === threadId) {
        setActive(remaining.length ? remaining[remaining.length - 1].threadId : null);
      }
      return remaining;
    });
  };

  const newThread = async () => {
    const t = await api.createThread("New chat");
    setThreads((prev) => [t, ...prev]);
    openThread(t);
    setMessagesByThread((prev) => ({ ...prev, [t.id]: [] }));
  };

  const deleteThread = async (threadId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    await api.deleteThread(threadId);
    setThreads((prev) => prev.filter((t) => t.id !== threadId));
    setOpenTabs((prev) => {
      const remaining = prev.filter((t) => t.threadId !== threadId);
      if (active === threadId) {
        setActive(remaining.length ? remaining[remaining.length - 1].threadId : null);
      }
      return remaining;
    });
    setMessagesByThread((prev) => {
      const copy = { ...prev };
      delete copy[threadId];
      return copy;
    });
  };

  const send = async () => {
    if (!input.trim() || streaming) return;
    if (!providerId && !selection) {
      setError("Select a model from the dropdown or download one from the Hub.");
      return;
    }
    if (!providerId && waitingForModel) {
      setError("Wait for the model to finish loading.");
      return;
    }
    if (!providerId && !modelReady && selection) {
      setSwitchingModel(true);
      setError(null);
      try {
        await activateModel(selection, models);
      } catch (e) {
        setError(e instanceof Error ? e.message : "Failed to load model into inference engine");
        return;
      } finally {
        setSwitchingModel(false);
        setLoadProgress(null);
      }
    }
    const content = input.trim();
    setInput("");
    setStreaming(true);
    setError(null);

    let threadId = active;
    if (!threadId) {
      const t = await api.createThread(content.slice(0, 48));
      threadId = t.id;
      setThreads((prev) => [t, ...prev]);
      openThread(t);
    }

    setMessagesByThread((prev) => ({
      ...prev,
      [threadId!]: [
        ...(prev[threadId!] ?? []),
        { id: crypto.randomUUID(), role: "user", content, created_at: new Date().toISOString() },
      ],
    }));

    let priorMessages = messagesByThread[threadId!] ?? [];
    if (priorMessages.length === 0 && threads.some((t) => t.id === threadId)) {
      priorMessages = await api.getMessages(threadId!);
      setMessagesByThread((prev) => ({ ...prev, [threadId!]: priorMessages }));
    }

    const history = [
      ...priorMessages.map((m) => ({ role: m.role, content: m.content })),
      { role: "user", content },
    ];

    const isOllamaOnly = selected?.kind === "ollama";
    const usingOllama = !providerId && effectiveBackend === "ollama";
    let assistantText = "";
    streamTextRef.current = "";
    streamThreadRef.current = threadId;

    const flushStreamText = () => {
      streamFlushRef.current = null;
      const text = streamTextRef.current;
      const tid = streamThreadRef.current;
      if (!tid) return;
      setMessagesByThread((prev) => {
        const list = [...(prev[tid] ?? [])];
        const last = list[list.length - 1];
        if (last?.role === "assistant") {
          list[list.length - 1] = { ...last, content: text };
        } else {
          list.push({
            id: crypto.randomUUID(),
            role: "assistant",
            content: text,
            created_at: new Date().toISOString(),
          });
        }
        return { ...prev, [tid]: list };
      });
    };

    try {
      const { promise, abort } = streamChat(
        {
          thread_id: threadId,
          messages: history,
          stream: true,
          tools: useTools && toolsAvailable,
          allow_code_exec: allowCodeExec && codeExecAvailable,
          provider_id: providerId || null,
          model_id: providerId || (usingOllama && isOllamaOnly) ? null : selection || null,
          ollama_model: usingOllama || isOllamaOnly ? selected?.ollama_model : null,
          inference_backend: providerId ? "auto" : effectiveBackend,
        },
        {
          onEvent: (event, data) => {
            if (event === "error") setError(data);
            if (event === "token" || event === "message") {
              if (event === "message") assistantText = data;
              else assistantText += data;
              streamTextRef.current = assistantText;
              if (streamFlushRef.current === null) {
                streamFlushRef.current = window.requestAnimationFrame(flushStreamText);
              }
            }
          },
        },
      );
      streamAbortRef.current = abort;
      await promise;
    } catch (e) {
      if (!(e instanceof DOMException && e.name === "AbortError")) {
        setError(e instanceof Error ? e.message : "Request failed");
      }
    } finally {
      if (streamFlushRef.current !== null) {
        window.cancelAnimationFrame(streamFlushRef.current);
        streamFlushRef.current = null;
      }
      if (streamTextRef.current) flushStreamText();
      streamThreadRef.current = null;
      streamAbortRef.current = null;
      setStreaming(false);
    }
  };

  const modelLabel = (m: InferenceModelOption) => {
    const engine = m.backend_labels[m.default_backend] || m.default_backend;
    const fit = m.hardware_fit_label ? ` · ${m.hardware_fit_label}` : "";
    return `${m.name} · ${engine}${fit}`;
  };

  return (
    <div className={`chat-app${sidebarOpen ? " chat-sidebar-open" : ""}`}>
      <button
        type="button"
        className="chat-sidebar-backdrop"
        onClick={() => setSidebarOpen(false)}
        aria-label="Close chat sidebar"
        tabIndex={sidebarOpen ? 0 : -1}
      />
      <aside className="chat-sidebar" aria-hidden={!sidebarOpen}>
        <div className={`chat-sidebar-header${waitingForModel ? " chat-sidebar-header-loading" : ""}`}>
          {!waitingForModel && (
            <button type="button" className="chat-new-btn" onClick={newThread}>
              <IconPlus size={16} />
              <span>New chat</span>
            </button>
          )}
          <button
            type="button"
            className="chat-sidebar-collapse"
            onClick={() => setSidebarOpen(false)}
            aria-label="Collapse chat sidebar"
            title="Collapse sidebar"
          >
            <IconChevronLeft size={18} strokeWidth={2.25} />
          </button>
        </div>
        <input
          className="chat-search"
          placeholder="Search chats…"
          value={threadSearch}
          onChange={(e) => setThreadSearch(e.target.value)}
        />
        <div className="chat-thread-list">
          {filteredThreads.map((t) => (
            <div
              key={t.id}
              className={`chat-thread-item${active === t.id ? " active" : ""}`}
              onClick={() => openThread(t)}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => e.key === "Enter" && openThread(t)}
            >
              <span className="chat-thread-title">{t.title}</span>
              <button
                type="button"
                className="chat-thread-delete"
                onClick={(e) => deleteThread(t.id, e)}
                aria-label="Delete chat"
              >
                <IconClose size={14} />
              </button>
            </div>
          ))}
        </div>
        <div className="chat-session-badge">
          <IconLock size={12} className="session-lock" />
          Encrypted session memory
          <span className="muted-text"> · clears on sign out</span>
        </div>
      </aside>

      <main className="chat-main">
        <header className="chat-topbar">
          <button
            type="button"
            className="chat-sidebar-toggle"
            onClick={() => setSidebarOpen((v) => !v)}
            aria-label={sidebarOpen ? "Collapse chat sidebar" : "Expand chat sidebar"}
            title={sidebarOpen ? "Collapse sidebar" : "Expand sidebar"}
          >
            {sidebarOpen ? (
              <IconChevronLeft size={18} strokeWidth={2.25} />
            ) : (
              <IconChevronRight size={18} strokeWidth={2.25} />
            )}
          </button>
          <ChatModelPicker
            models={models}
            selection={selection}
            disabled={!!providerId}
            switching={switchingModel}
            modelLabel={modelLabel}
            onSelectLocal={handleModelChange}
            onSelectCatalog={handleCatalogSelect}
          />
          <button
            type="button"
            className="chat-refresh-models"
            onClick={refreshModels}
            title="Refresh model list"
            aria-label="Refresh models"
          >
            <IconRefresh size={15} />
          </button>
          {selected && backendOptions.length > 1 && !providerId && (
            <select
              className="chat-engine-select"
              value={inferenceBackend}
              onChange={async (e) => {
                const next = e.target.value;
                if (next !== inferenceBackend) {
                  streamAbortRef.current?.();
                  streamAbortRef.current = null;
                  if (streaming) setStreaming(false);
                  try {
                    await api.cancelInference();
                    setLoadedModelId(null);
                    setLoadedBackend(null);
                  } catch {
                    /* ignore */
                  }
                }
                setInferenceBackend(next);
                userPickedBackendRef.current = true;
                if (selection && !providerId) {
                  const model = models.find((m) => m.id === selection);
                  try {
                    setSwitchingModel(true);
                    setLoadProgress(
                      initialLoadProgress(model?.name || "model", model?.size_bytes ?? 0),
                    );
                    const loaded = await preloadWithProgress(selection, next, setLoadProgress);
                    setLoadedModelId(selection);
                    setLoadedBackend(loaded);
                  } catch (e) {
                    setLoadedModelId(null);
                    setLoadedBackend(null);
                    setError(e instanceof Error ? e.message : "Failed to load model into inference engine");
                  } finally {
                    setSwitchingModel(false);
                    setLoadProgress(null);
                  }
                }
              }}
            >
              {backendOptions.map((b) => (
                <option key={b} value={b}>{selected.backend_labels?.[b] || BACKEND_LABELS[b] || b}</option>
              ))}
            </select>
          )}
          {providers.length > 0 ? (
            <select
              className="chat-provider-select"
              value={providerId}
              onChange={(e) => void handleProviderChange(e.target.value)}
            >
              <option value="">Local</option>
              {providers.map((p) => (
                <option key={p.id} value={p.id}>{p.name}</option>
              ))}
            </select>
          ) : (
            <span className="chat-provider-label muted-text">Local</span>
          )}
          <label className="chat-tools-toggle">
            <input type="checkbox" checked={useTools} disabled={!toolsAvailable} onChange={(e) => setUseTools(e.target.checked)} />
            Tools
          </label>
          {switchingModel && !loadProgress && (
            <span className="chat-vram-hint">Preparing model…</span>
          )}
          {streaming && !providerId && (
            <span className="chat-vram-hint muted-text">Generating — switch model anytime</span>
          )}
          {selected && selectedFit && !providerId && (
            <HardwareFitBadge fit={selectedFit} label={selected.hardware_fit_label} />
          )}
          {hwProfile?.tier_label && !providerId && (
            <span className="chat-hw-tier muted-text" title={hwProfile.privacy}>{hwProfile.tier_label}</span>
          )}
          {useTools && codeExecAvailable && (
            <label className="chat-tools-toggle chat-tools-warn">
              <input type="checkbox" checked={allowCodeExec} onChange={(e) => setAllowCodeExec(e.target.checked)} />
              Code
            </label>
          )}
        </header>

        {showModelStatus && (
          <div className="chat-model-status">
            {(selected || (pendingRepo && !modelReady)) && (
              <div className="chat-model-status-selected">
                <span className="chat-model-status-label">
                  {effectiveLoadProgress?.phase === "download" ? "Downloading from Hugging Face" : "Selected model"}
                </span>
                <span className="chat-model-status-name">{selected?.name || pendingModelLabel}</span>
                {pendingRepo && !modelReady && (
                  <span className="chat-model-status-engine muted-text">{pendingRepo}</span>
                )}
                {selected && (
                  <span className="chat-model-status-engine muted-text">
                    {selected.backend_labels?.[effectiveBackend] || BACKEND_LABELS[effectiveBackend] || effectiveBackend}
                  </span>
                )}
                {modelReady && !effectiveLoadProgress && (
                  <span className="chat-model-status-ready">
                    Loaded in {BACKEND_LABELS[effectiveBackend] || effectiveBackend}
                  </span>
                )}
              </div>
            )}
            {effectiveLoadProgress && (
              <ModelLoadProgress
                progress={effectiveLoadProgress}
                modelName={selected?.name || pendingModelLabel}
              />
            )}
          </div>
        )}

        {openTabs.length > 1 && (
          <div className="chat-tabs">
            {openTabs.map((tab) => (
              <button
                key={tab.threadId}
                type="button"
                className={`chat-tab${active === tab.threadId ? " active" : ""}`}
                onClick={() => setActive(tab.threadId)}
              >
                <span className="chat-tab-title">{tab.title.slice(0, 24)}</span>
                <span className="chat-tab-close" onClick={(e) => closeTab(tab.threadId, e)} aria-label="Close tab">
                  <IconClose size={12} />
                </span>
              </button>
            ))}
          </div>
        )}

        <div className="chat-messages-wrap">
          {!active && messages.length === 0 ? (
            <div className="chat-empty">
              <div className="chat-empty-icon">
                <IconAssistant size={32} />
              </div>
              <h2>How can I help you today?</h2>
              {waitingForModel ? (
                <p className="muted-text">Model loading — chat will be ready shortly.</p>
              ) : (
                <>
                  <p>Start a new chat — your conversation is encrypted locally until you sign out.</p>
                  <button type="button" className="btn btn-primary" onClick={newThread}>New chat</button>
                </>
              )}
            </div>
          ) : (
            <div className="chat-messages">
              {messages.map((m) => (
                <div key={m.id} className={`chat-bubble chat-bubble-${m.role}`}>
                  <div className="chat-avatar">
                    {m.role === "user" ? (
                      <span className="chat-avatar-text">You</span>
                    ) : (
                      <IconAssistant size={14} />
                    )}
                  </div>
                  <div className="chat-bubble-content">{m.content}</div>
                </div>
              ))}
              {streaming && messages[messages.length - 1]?.role !== "assistant" && (
                <div className="chat-bubble chat-bubble-assistant">
                  <div className="chat-avatar">
                    <IconAssistant size={14} />
                  </div>
                  <div className="chat-bubble-content chat-typing"><span /><span /><span /></div>
                </div>
              )}
              <div ref={bottomRef} />
            </div>
          )}
          {error && (
            <p className="chat-error">
              {error}
              {(error.includes("gated") || error.includes("Access denied") || error.toLowerCase().includes("token")) && (
                <>
                  {" "}
                  <a href="/settings?tab=huggingface">Open Hugging Face settings</a>
                </>
              )}
            </p>
          )}
          {selected?.hardware_fit === "unlikely" && !providerId && (
            <p className="chat-hw-warn">{selected.hardware_note || "This model may exceed available memory on your machine."}</p>
          )}
        </div>

        <footer className="chat-composer-wrap">
          <div className="chat-composer">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Message Seiso…"
              rows={1}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  send();
                }
              }}
            />
            <button
              type="button"
              className="chat-send-btn"
              onClick={send}
              disabled={streaming || waitingForModel || !input.trim()}
              aria-label="Send message"
            >
              <IconSend size={16} />
            </button>
          </div>
          <p className="chat-composer-hint">Shift+Enter for new line · Memory encrypted in local DB</p>
        </footer>
      </main>
    </div>
  );
}
