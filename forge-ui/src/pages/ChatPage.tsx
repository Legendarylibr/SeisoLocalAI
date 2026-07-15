import { memo, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { api, ChatMessage, ChatThread, CatalogModel, InferenceModelOption, streamChat, VramStatus } from "@/lib/api";
import { usePlatformSettings } from "@/context/PlatformSettingsContext";
import { bootstrapChatSession, CHAT_BACKEND_STORAGE_KEY, CHAT_MODEL_STORAGE_KEY, hasChatNavTarget, initializeChatSession, isChatModelReady, modelMemoryBlocked, modelMemoryBlockReason, needsHubDownload, preloadWithProgress, resolveInferenceBackend } from "@/lib/chatModel";
import { hasLoadedInferenceMemory } from "@/lib/hubHardware";
import { invalidateApiCache } from "@/lib/api/getCache";
import { useHardwareProfile } from "@/hooks/useHardware";
import { writeStoredModel } from "@/lib/modelSelection";
import { createStreamDisplaySink } from "@/lib/streamDisplay";
import {
  computeTokensPerSec,
  formatTokensPerSec,
  parseStreamStats,
  resolveOutputTokenCount,
} from "@/lib/streamSpeed";
import type { ChatContextStatus } from "@/lib/api/types";
import { ROUTER_MODEL_ID } from "@/lib/api/types";
import { ModelProgressState, initialDownloadProgress, initialLoadProgress } from "@/lib/modelProgress";
import { ChatBubble } from "@/components/ChatBubble";
import { ChatModelPicker } from "@/components/ChatModelPicker";
import { ChatContextBar } from "@/components/ChatContextBar";
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
  IconEject,
  IconSend,
} from "@/components/Icons";

const BACKEND_LABELS: Record<string, string> = {
  llamacpp: "llama.cpp",
  llamaswap: "Ollama sidecar",
  mlx: "MLX",
  torch: "PyTorch",
};

function resolveBackendLabel(
  backend: string,
  labels: Record<string, string>,
  optionLabels?: Record<string, string>,
): string {
  return optionLabels?.[backend] || labels[backend] || BACKEND_LABELS[backend] || backend;
}

function showStreamingTyping(el: HTMLElement) {
  el.classList.add("chat-typing");
  el.replaceChildren();
  for (let i = 0; i < 3; i += 1) {
    el.appendChild(document.createElement("span"));
  }
}

function showStreamingText(el: HTMLElement, text: string) {
  el.classList.remove("chat-typing");
  el.textContent = text;
}

/** Isolated from ChatPage re-renders so imperative stream text is not wiped. */
const StreamingBubble = memo(function StreamingBubble({
  contentRef,
}: {
  contentRef: React.RefObject<HTMLDivElement | null>;
}) {
  return (
    <div className="chat-bubble chat-bubble-assistant chat-bubble-streaming">
      <div className="chat-avatar">
        <IconAssistant size={14} />
      </div>
      <div className="chat-bubble-content chat-typing" ref={contentRef}>
        <span />
        <span />
        <span />
      </div>
    </div>
  );
});

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
  const [streamTps, setStreamTps] = useState<number | null>(null);
  const [lastTps, setLastTps] = useState<number | null>(null);
  /** Message ids whose reply still hit max length after auto-continue. */
  const [truncatedMessageIds, setTruncatedMessageIds] = useState<Record<string, true>>({});
  const [error, setError] = useState<string | null>(null);
  const [threadSearch, setThreadSearch] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [useTools, setUseTools] = useState(false);
  const [allowCodeExec, setAllowCodeExec] = useState(false);
  const [knowledgeBases, setKnowledgeBases] = useState<Array<{ id: string; chunk_count: number; has_index: boolean }>>([]);
  const [knowledgeBaseId, setKnowledgeBaseId] = useState("");
  const [contextStatus, setContextStatus] = useState<ChatContextStatus | null>(null);
  const [contextLoading, setContextLoading] = useState(false);
  const [providerId, setProviderId] = useState("");
  const [selection, setSelection] = useState("");
  const [inferenceBackend, setInferenceBackend] = useState("auto");
  const [models, setModels] = useState<InferenceModelOption[]>([]);
  const [providers, setProviders] = useState<Array<{ id: string; name: string; provider_type: string }>>([]);
  const { settings } = usePlatformSettings();
  const security = settings?.security ?? null;
  const { profile: hwProfile, refresh: refreshHwProfile } = useHardwareProfile();
  const backendLabels = hwProfile?.inference_backend_labels ?? {};
  const [switchingModel, setSwitchingModel] = useState(false);
  const [loadProgress, setLoadProgress] = useState<ModelProgressState | null>(null);
  const [loadedModelId, setLoadedModelId] = useState<string | null>(null);
  const [loadedBackend, setLoadedBackend] = useState<string | null>(null);
  const [vramStatus, setVramStatus] = useState<VramStatus | null>(null);
  const [, setRouterStatus] = useState<Record<string, unknown> | null>(null);
  const [freeingMemory, setFreeingMemory] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const scrollFrameRef = useRef<number | null>(null);
  const streamAbortRef = useRef<(() => void) | null>(null);
  const streamingElRef = useRef<HTMLDivElement>(null);
  const streamDisplayRef = useRef<ReturnType<typeof createStreamDisplaySink> | null>(null);
  const streamThreadRef = useRef<string | null>(null);
  const genStartRef = useRef<number | null>(null);
  const outputTokensRef = useRef(0);
  const tpsFrameRef = useRef<number | null>(null);
  const pendingTpsRef = useRef<number | null>(null);
  const userPickedBackendRef = useRef(false);
  const bootstrapGenRef = useRef(0);
  const bootstrapAbortRef = useRef<AbortController | null>(null);
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
  // Backend auto-selects context + completion budget (no UI inference presets).
  const chatBackend = useMemo(
    () => (providerId ? "auto" : effectiveBackend),
    [providerId, effectiveBackend],
  );
  const autoMaxTokens = useMemo(() => {
    // Desired overall reply length (backend OOM-clamps each chunk and auto-continues).
    const recommended = selected?.recommended_max_tokens ?? 2048;
    return Math.max(1, Math.min(131072, recommended));
  }, [selected?.recommended_max_tokens]);
  const isRouterMode = selected?.kind === "router" || selection === ROUTER_MODEL_ID;
  const modelReady = useMemo(
    () => isChatModelReady(selection, chatBackend, loadedModelId, loadedBackend, selected?.kind),
    [selection, chatBackend, loadedModelId, loadedBackend, selected?.kind],
  );
  const showModelStatus =
    !isRouterMode &&
    !providerId &&
    !!(loadProgress || switchingModel || (hasNavTarget && !modelReady) || (selected && !modelReady));
  const waitingForModel = switchingModel || !!loadProgress;
  const effectiveLoadProgress =
    loadProgress ??
    (switchingModel && models.length > 0 && hasNavTarget && pendingRepo && needsHubDownload(models, navTarget)
      ? initialDownloadProgress(pendingRepo, pendingDownloadBytes)
      : null);
  const selectedFit = selected?.hardware_fit;
  const modelBlocked = !providerId && modelMemoryBlocked(selected, hwProfile?.vram_headroom_mb);

  const memoryBlockHint = useCallback(
    (model: InferenceModelOption | CatalogModel | null | undefined) => {
      let reason = modelMemoryBlockReason(model);
      if (hasLoadedInferenceMemory(vramStatus)) {
        reason = `${reason} Free memory first, then retry.`;
      }
      return reason;
    },
    [vramStatus],
  );

  const modelBlockReason = memoryBlockHint(selected);

  const refreshVramStatus = useCallback(async () => {
    try {
      const status = await api.vramStatus();
      setVramStatus(status);
      return status;
    } catch {
      return null;
    }
  }, []);

  const showFreeMemory =
    hasLoadedInferenceMemory(vramStatus) || Boolean(loadedModelId);

  const filteredThreads = useMemo(() => {
    const q = threadSearch.toLowerCase();
    if (!q) return threads;
    return threads.filter((t) => t.title.toLowerCase().includes(q));
  }, [threads, threadSearch]);

  const refreshModels = useCallback(async () => {
    const r = await api.listInferenceModels();
    setModels(r.models);
    if (selection) {
      invalidateApiCache(`/inference/models/${selection}/variants`);
    }
    if (selection) {
      const still = r.models.find((m) => m.id === selection);
      if (still) return r.models;
    }
    const pick = r.models.find((m) => !modelMemoryBlocked(m) && (m.hardware_fit === "ideal" || m.hardware_fit === "good"))?.id
      || r.models.find((m) => !modelMemoryBlocked(m))?.id
      || r.models[0]?.id;
    if (pick) {
      const model = r.models.find((m) => m.id === pick);
      setSelection(pick);
      setInferenceBackend(resolveInferenceBackend(model ?? null, hwProfile));
    }
    return r.models;
  }, [selection, hwProfile]);

  const releaseInferenceMemory = useCallback(async () => {
    const status = await api.freeMemory();
    setVramStatus(status);
    setLoadedModelId(null);
    setLoadedBackend(null);
    await Promise.all([refreshHwProfile(), refreshModels()]);
    return status;
  }, [refreshHwProfile, refreshModels]);

  const preloadInferenceOptions = useCallback(
    (modelId: string, list: InferenceModelOption[]) => {
      const model = list.find((m) => m.id === modelId) ?? null;
      const maxTokens = Math.max(1, Math.min(131072, model?.recommended_max_tokens ?? 2048));
      // Always auto-context: backend sizes + pins KV after preload (no UI presets).
      return { maxTokens, nCtx: null as number | null };
    },
    [],
  );

  const activateModel = useCallback(
    async (modelId: string, list: InferenceModelOption[], backendOverride?: string) => {
      const next = list.find((m) => m.id === modelId);
      if (!next) {
        throw new Error("Model not found in inventory after download");
      }
      if (modelMemoryBlocked(next, hwProfile?.vram_headroom_mb)) {
        throw new Error(modelMemoryBlockReason(next));
      }
      const preloadOpts = preloadInferenceOptions(modelId, list);
      setSelection(modelId);
      writeStoredModel(CHAT_MODEL_STORAGE_KEY, modelId);
      const backend =
        backendOverride ??
        resolveInferenceBackend(next, hwProfile, userPickedBackendRef.current ? inferenceBackend : undefined);
      setInferenceBackend(backend);
      if (next.kind === "router") {
        setLoadedModelId(modelId);
        setLoadedBackend("router");
        writeStoredModel(CHAT_BACKEND_STORAGE_KEY, "router");
        api.routerStatus().then((s) => setRouterStatus(s.detail ?? s)).catch(() => setRouterStatus(null));
        return "router";
      }
      if (providerId) return backend;
      setLoadProgress(initialLoadProgress(next.name, next.size_bytes));
      const loaded = await preloadWithProgress(modelId, backend, setLoadProgress, undefined, {
        maxTokens: preloadOpts.maxTokens,
        nCtx: preloadOpts.nCtx,
      });
      setLoadedModelId(modelId);
      setLoadedBackend(loaded.backend);
      writeStoredModel(CHAT_BACKEND_STORAGE_KEY, loaded.backend);
      return loaded.backend;
    },
    [providerId, hwProfile, inferenceBackend, preloadInferenceOptions],
  );

  const handleModelChange = async (modelId: string) => {
    const next = models.find((m) => m.id === modelId);
    if (modelMemoryBlocked(next, hwProfile?.vram_headroom_mb)) {
      setError(memoryBlockHint(next));
      return;
    }
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
        await releaseInferenceMemory();
      } catch {
        /* best-effort memory release */
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

  const handleCatalogSelect = (model: CatalogModel) => {
    if (modelMemoryBlocked(model, hwProfile?.vram_headroom_mb)) {
      setError(memoryBlockHint(model));
      return;
    }
    bootstrapAbortRef.current?.abort();
    streamAbortRef.current?.();
    streamAbortRef.current = null;
    if (streaming) setStreaming(false);
    setError(null);
    const params = new URLSearchParams({ repo: model.repo_id });
    if (model.download_bytes && model.download_bytes > 0) {
      params.set("bytes", String(model.download_bytes));
    }
    setSearchParams(params, { replace: true });
  };

  const handleCancelModelLoad = useCallback(() => {
    bootstrapAbortRef.current?.abort();
    bootstrapAbortRef.current = null;
    setSwitchingModel(false);
    setLoadProgress(null);
    if (pendingRepo) {
      setSearchParams(selection ? { model: selection } : {}, { replace: true });
    }
  }, [pendingRepo, selection, setSearchParams]);

  const handleFreeMemory = useCallback(async () => {
    if (!showFreeMemory || freeingMemory) return;
    streamAbortRef.current?.();
    streamAbortRef.current = null;
    if (streaming) setStreaming(false);
    setFreeingMemory(true);
    setError(null);
    try {
      await releaseInferenceMemory();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to free memory");
    } finally {
      setFreeingMemory(false);
    }
  }, [showFreeMemory, freeingMemory, streaming, releaseInferenceMemory]);

  const handleProviderChange = async (nextProvider: string) => {
    if (nextProvider === providerId) return;
    streamAbortRef.current?.();
    streamAbortRef.current = null;
    if (streaming) setStreaming(false);
    if (nextProvider && (showFreeMemory || loadedModelId)) {
      try {
        await releaseInferenceMemory();
      } catch {
        /* ignore */
      }
    }
    if (!nextProvider && providerId) {
      try {
        await releaseInferenceMemory();
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
          const preloadOpts = preloadInferenceOptions(selection, models);
          const loaded = await preloadWithProgress(
            selection,
            chatBackend,
            setLoadProgress,
            undefined,
            { maxTokens: preloadOpts.maxTokens, nCtx: preloadOpts.nCtx },
          );
          setLoadedModelId(selection);
          setLoadedBackend(loaded.backend);
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
    api.listKnowledgeBases()
      .then((res) => setKnowledgeBases(res.bases.filter((b) => b.has_index && b.chunk_count > 0)))
      .catch(() => {});
    void refreshVramStatus();
  }, [refreshVramStatus]);

  useEffect(() => {
    if (!pendingRepo && sessionInitRef.current && (!pendingModel || pendingModel === selection)) {
      return;
    }

    const bootstrapGen = ++bootstrapGenRef.current;
    bootstrapAbortRef.current?.abort();
    const controller = new AbortController();
    bootstrapAbortRef.current = controller;
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
        const [initialModelsResp] = await Promise.all([
          api.listInferenceModels(),
        ]);
        if (cancelled) return;

        const initialModels = initialModelsResp.models;
        const commonOptions = {
          preload: !providerId,
          providerActive: !!providerId,
          onProgress: setLoadProgress,
          initialModels,
          hwProfile,
          signal: controller.signal,
          maxTokens: autoMaxTokens,
        };

        if (pendingRepo) {
          if (needsHubDownload(initialModels, navTarget)) {
            setLoadProgress(initialDownloadProgress(pendingRepo, pendingDownloadBytes));
          } else {
            setLoadProgress(null);
          }
          const result = await bootstrapChatSession(navTarget, commonOptions);
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
          const result = await bootstrapChatSession({ modelId: pendingModel }, commonOptions);
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
      } finally {
        if (bootstrapAbortRef.current === controller) {
          bootstrapAbortRef.current = null;
        }
        if (bootstrapGen === bootstrapGenRef.current) {
          setSwitchingModel(false);
          setLoadProgress(null);
        }
      }
    };

    void bootstrap();
    return () => {
      cancelled = true;
      controller.abort();
      if (bootstrapAbortRef.current === controller) {
        bootstrapAbortRef.current = null;
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pendingModel, pendingRepo, pendingDownloadBytes]);

  useEffect(() => () => {
    sessionInitRef.current = false;
  }, []);

  useEffect(() => {
    if (active && !messagesByThread[active]) loadMessages(active).catch(console.error);
  }, [active, messagesByThread, loadMessages]);

  const scrollToBottom = useCallback((behavior: ScrollBehavior = "auto") => {
    if (scrollFrameRef.current !== null) {
      window.cancelAnimationFrame(scrollFrameRef.current);
    }
    scrollFrameRef.current = window.requestAnimationFrame(() => {
      scrollFrameRef.current = null;
      bottomRef.current?.scrollIntoView({ behavior });
    });
  }, []);

  useEffect(() => {
    scrollToBottom(streaming ? "auto" : "smooth");
  }, [messages, active, streaming, scrollToBottom]);

  useEffect(() => () => {
    if (scrollFrameRef.current !== null) {
      window.cancelAnimationFrame(scrollFrameRef.current);
    }
    if (tpsFrameRef.current !== null) {
      window.cancelAnimationFrame(tpsFrameRef.current);
    }
  }, []);

  const messageCount = messages.length;

  useEffect(() => {
    // Context is derived from message history; during streaming the assistant
    // message updates every animation frame (~60/s), which would exceed the
    // global API rate limit (240 req/min on localhost).
    if (streaming) return;

    let cancelled = false;
    const timeout = window.setTimeout(() => {
      const refreshContext = async () => {
        setContextLoading(true);
        try {
          const status = await api.getContextStatus({
            thread_id: active,
            max_tokens: autoMaxTokens,
            n_ctx: null,
            tools: useTools && toolsAvailable,
            knowledge_base_id: knowledgeBaseId || null,
            model_id: providerId ? null : selection || null,
            draft_message: input.trim() || null,
          });
          if (!cancelled) setContextStatus(status);
        } catch {
          if (!cancelled) setContextStatus(null);
        } finally {
          if (!cancelled) setContextLoading(false);
        }
      };
      void refreshContext();
    }, 400);

    return () => {
      cancelled = true;
      window.clearTimeout(timeout);
    };
  }, [
    active,
    messageCount,
    input,
    streaming,
    autoMaxTokens,
    knowledgeBaseId,
    selection,
    providerId,
    useTools,
    toolsAvailable,
  ]);

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
    if (!providerId && selection && !effectiveBackend) {
      const installHint = selected?.install_hints?.[0];
      setError(
        selected?.format === "gguf"
          ? installHint
            ? `GGUF chat needs llama.cpp. Run: ${installHint}`
            : 'GGUF chat needs llama-cpp-python. Run: pip install "llama-cpp-python>=0.3" or re-run start.'
          : "No installed local inference engine can load this model.",
      );
      return;
    }
    if (!providerId && !modelReady && selection) {
      setSwitchingModel(true);
      setError(null);
      try {
        await activateModel(selection, models, chatBackend);
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

    let displayedMessages = messagesByThread[threadId!] ?? [];
    if (displayedMessages.length === 0 && threads.some((t) => t.id === threadId)) {
      displayedMessages = await api.getMessages(threadId!);
    }

    setMessagesByThread((prev) => ({
      ...prev,
      [threadId!]: [
        ...displayedMessages,
        { id: crypto.randomUUID(), role: "user", content, created_at: new Date().toISOString() },
      ],
    }));

    const latestUserMessage = [{ role: "user", content }];

    let assistantText = "";
    let progressText = "";
    let streamFailed = false;
    let replyTruncated = false;
    streamThreadRef.current = threadId;
    genStartRef.current = null;
    outputTokensRef.current = 0;
    setStreamTps(null);

    const commitAssistantMessage = (text: string, truncated = false) => {
      const tid = streamThreadRef.current;
      if (!tid || !text.trim()) return;
      const id = crypto.randomUUID();
      setMessagesByThread((prev) => ({
        ...prev,
        [tid]: [
          ...(prev[tid] ?? []),
          {
            id,
            role: "assistant" as const,
            content: text,
            created_at: new Date().toISOString(),
          },
        ],
      }));
      if (truncated) {
        setTruncatedMessageIds((prev) => ({ ...prev, [id]: true }));
      }
    };

    const flushTpsUpdate = (finalize = false) => {
      if (tpsFrameRef.current !== null) {
        window.cancelAnimationFrame(tpsFrameRef.current);
        tpsFrameRef.current = null;
      }
      if (finalize) {
        pendingTpsRef.current = null;
        setStreamTps(null);
        return;
      }
      const pending = pendingTpsRef.current;
      if (pending !== null) setStreamTps(pending);
    };

    const scheduleTpsUpdate = (finalize = false) => {
      const tokenCount = resolveOutputTokenCount(outputTokensRef.current, assistantText);
      if (tokenCount <= 0) return;
      const now = performance.now();
      if (genStartRef.current === null) genStartRef.current = now;
      const tps = computeTokensPerSec(tokenCount, now - genStartRef.current);
      if (tps === null) return;
      if (finalize) {
        flushTpsUpdate(true);
        setLastTps(tps);
        return;
      }
      pendingTpsRef.current = tps;
      if (tpsFrameRef.current !== null) return;
      tpsFrameRef.current = window.requestAnimationFrame(() => {
        tpsFrameRef.current = null;
        flushTpsUpdate();
      });
    };

    const streamDisplay = createStreamDisplaySink(
      (text) => {
        const el = streamingElRef.current;
        if (!el) return;
        if (text) showStreamingText(el, text);
        else showStreamingTyping(el);
      },
      () => scrollToBottom("auto"),
    );
    streamDisplayRef.current = streamDisplay;
    streamDisplay.reset();
    if (streamingElRef.current) {
      showStreamingTyping(streamingElRef.current);
    }

    try {
      const { promise, abort } = streamChat(
        {
          thread_id: threadId,
          messages: latestUserMessage,
          stream: true,
          tools: useTools && toolsAvailable,
          allow_code_exec: allowCodeExec && codeExecAvailable,
          provider_id: providerId || null,
          model_id: providerId ? null : selection || null,
          inference_backend: providerId ? "auto" : chatBackend,
          // max_tokens / n_ctx: backend auto-clamps and auto-continues length stops.
          max_tokens: autoMaxTokens,
          temperature: 0.7,
          knowledge_base_id: knowledgeBaseId || null,
        },
        {
          onEvent: (event, data) => {
            if (event === "error") {
              streamFailed = true;
              setError(data);
              return;
            }
            if (event === "log") {
              // Auto-continue logs are server status, not model output.
              if (/Reply hit max length/i.test(data)) {
                return;
              }
              progressText = `${progressText}${progressText ? "\n" : ""}${data}`;
              if (!assistantText) {
                streamDisplay.push(progressText);
              }
              return;
            }
            if (event === "stats") {
              const stats = parseStreamStats(data);
              if (stats) {
                outputTokensRef.current = stats.output_tokens;
                if (typeof stats.truncated === "boolean") {
                  replyTruncated = stats.truncated;
                }
              }
              scheduleTpsUpdate();
              return;
            }
            if (event === "done") {
              scheduleTpsUpdate(true);
              return;
            }
            if (event === "token" || event === "message") {
              if (event === "message") assistantText = data;
              else assistantText += data;
              streamDisplay.push(assistantText);
              scheduleTpsUpdate();
            }
          },
        },
      );
      streamAbortRef.current = abort;
      await promise;
    } catch (e) {
      if (!(e instanceof DOMException && e.name === "AbortError")) {
        streamFailed = true;
        setError(e instanceof Error ? e.message : "Request failed");
      }
    } finally {
      streamDisplayRef.current?.flush();
      streamDisplayRef.current?.reset();
      streamDisplayRef.current = null;
      if (streamingElRef.current) {
        streamingElRef.current.textContent = "";
      }
      if (!streamFailed && assistantText.trim()) {
        commitAssistantMessage(assistantText, replyTruncated);
      }
      if (assistantText.trim() && genStartRef.current !== null) {
        const tokenCount = resolveOutputTokenCount(outputTokensRef.current, assistantText);
        const tps = computeTokensPerSec(tokenCount, performance.now() - genStartRef.current);
        if (tps !== null) setLastTps(tps);
      }
      genStartRef.current = null;
      outputTokensRef.current = 0;
      streamThreadRef.current = null;
      streamAbortRef.current = null;
      flushTpsUpdate(true);
      setStreaming(false);
    }
  };

  const stopStreaming = () => {
    streamAbortRef.current?.();
    streamAbortRef.current = null;
    setStreaming(false);
    setError(null);
  };

  const modelLabel = (m: InferenceModelOption) => {
    const engine = m.default_backend
      ? m.backend_labels[m.default_backend] || m.default_backend
      : m.format === "gguf"
        ? m.install_hints?.length
          ? "llama.cpp missing"
          : "missing llama.cpp"
        : "missing local runtime";
    const fit = m.hardware_fit_label ? ` · ${m.hardware_fit_label}` : "";
    return `${m.name} · ${engine}${fit}`;
  };

  const displayedTps = streaming ? streamTps : lastTps;

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
          <div className="chat-model-controls">
            <ChatModelPicker
              models={models}
              selection={selection}
              disabled={!!providerId}
              switching={switchingModel}
              headroomMb={hwProfile?.vram_headroom_mb}
              modelLabel={modelLabel}
              onSelectLocal={handleModelChange}
              onSelectCatalog={handleCatalogSelect}
            />
            {showFreeMemory && (
              <button
                type="button"
                className="chat-free-memory"
                onClick={() => void handleFreeMemory()}
                disabled={freeingMemory || switchingModel}
                title="Free memory — unload model from RAM/VRAM (keeps selection)"
                aria-label="Free memory"
              >
                <IconEject size={15} />
              </button>
            )}
          </div>
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
                    await releaseInferenceMemory();
                  } catch {
                    /* ignore */
                  }
                }
                setInferenceBackend(next);
                writeStoredModel(CHAT_BACKEND_STORAGE_KEY, next);
                userPickedBackendRef.current = true;
                if (selection && !providerId) {
                  const model = models.find((m) => m.id === selection);
                  try {
                    setSwitchingModel(true);
                    setLoadProgress(
                      initialLoadProgress(model?.name || "model", model?.size_bytes ?? 0),
                    );
                    const preloadOpts = preloadInferenceOptions(selection, models);
                    const loaded = await preloadWithProgress(
                      selection,
                      next,
                      setLoadProgress,
                      undefined,
                      { maxTokens: preloadOpts.maxTokens, nCtx: preloadOpts.nCtx },
                    );
                    setLoadedModelId(selection);
                    setLoadedBackend(loaded.backend);
                    writeStoredModel(CHAT_BACKEND_STORAGE_KEY, loaded.backend);
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
                <option key={b} value={b}>{resolveBackendLabel(b, backendLabels, selected.backend_labels)}</option>
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
          {knowledgeBases.length > 0 && (
            <select
              className="chat-knowledge-select"
              value={knowledgeBaseId}
              onChange={(e) => setKnowledgeBaseId(e.target.value)}
              title="Inject knowledge base context into chat"
            >
              <option value="">No knowledge base</option>
              {knowledgeBases.map((kb) => (
                <option key={kb.id} value={kb.id}>
                  KB: {kb.id} ({kb.chunk_count})
                </option>
              ))}
            </select>
          )}
          <label className="chat-tools-toggle">
            <input type="checkbox" checked={useTools} disabled={!toolsAvailable} onChange={(e) => setUseTools(e.target.checked)} />
            Tools
          </label>
          {isRouterMode && !providerId && (
            <span className="chat-vram-hint" title="Routes to vLLM specialists via classifier + RL policy">
              Smart Router · auto-route
            </span>
          )}
          {switchingModel && !loadProgress && !isRouterMode && (
            <span className="chat-vram-hint">Preparing model…</span>
          )}
          {streaming && !providerId && (
            <span className="chat-vram-hint muted-text">Generating — switch model anytime</span>
          )}
          {streaming && (
            <button type="button" className="btn" onClick={stopStreaming}>
              Stop
            </button>
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

        <ChatContextBar
          status={contextStatus}
          loading={contextLoading}
        />

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
                    {effectiveBackend
                      ? resolveBackendLabel(effectiveBackend, backendLabels, selected.backend_labels)
                      : selected.format === "gguf"
                        ? "Missing Ollama sidecar"
                        : "Missing local inference runtime"}
                  </span>
                )}
                {modelReady && !effectiveLoadProgress && (
                  <span className="chat-model-status-ready">
                    Loaded in {resolveBackendLabel(effectiveBackend, backendLabels, selected?.backend_labels)}
                  </span>
                )}
              </div>
            )}
            {effectiveLoadProgress && (
              <ModelLoadProgress
                progress={effectiveLoadProgress}
                modelName={selected?.name || pendingModelLabel}
                onCancel={waitingForModel ? handleCancelModelLoad : undefined}
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
                <ChatBubble
                  key={m.id}
                  message={m}
                  truncated={Boolean(truncatedMessageIds[m.id])}
                />
              ))}
              {streaming && <StreamingBubble contentRef={streamingElRef} />}
              <div ref={bottomRef} />
            </div>
          )}
          {error && (
            <p className="chat-error">
              {error}
              {(error.includes("gated") || error.includes("Access denied") || error.toLowerCase().includes("token") || error.toLowerCase().includes("csrf")) && (
                <>
                  {" "}
                  <a href="/settings?tab=huggingface">Open Hugging Face settings</a>
                </>
              )}
            </p>
          )}
          {modelBlocked && (
            <p className="chat-hw-warn">{modelBlockReason}</p>
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
              disabled={streaming || waitingForModel || modelBlocked || !input.trim()}
              aria-label="Send message"
            >
              <IconSend size={16} />
            </button>
          </div>
          <div className="chat-composer-meta">
            {displayedTps != null && (
              <span className="chat-throughput" aria-live="polite">
                {formatTokensPerSec(displayedTps)}
              </span>
            )}
            <p className="chat-composer-hint">Shift+Enter for new line · Memory encrypted in local DB</p>
          </div>
        </footer>
      </main>
    </div>
  );
}
