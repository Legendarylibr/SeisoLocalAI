import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { api, GuideStep } from "@/lib/api";
import { useLiveMetrics } from "@/context/MetricsContext";
import { useHardwareProfile } from "@/hooks/useHardware";
import { chatPath } from "@/lib/chatModel";
import { PageHeader } from "@/components/PageHeader";
import {
  IconChat,
  IconTrain,
  IconCompress,
  IconChevronRight,
  IconCpu,
  IconMemory,
  IconGpu,
  IconQuant,
  IconRecipes,
  IconKnowledge,
} from "@/components/Icons";
import { PipelineStrip } from "@/components/research/PipelineStrip";

const GOALS = [
  { id: "chat", label: "Chat & Inference", path: "/chat", Icon: IconChat, desc: "Run GGUF, MLX, or PyTorch models locally with encrypted sessions" },
  { id: "train", label: "Train & Finetune", path: "/train", Icon: IconTrain, desc: "Fine-tune with LoRA on your hardware" },
  { id: "compress", label: "Compress", path: "/compress", Icon: IconCompress, desc: "Quantize and shrink models for faster inference" },
] as const;

const RESEARCH_PIPELINES = [
  {
    id: "rl-quant",
    title: "RL Quantization",
    desc: "Reward-guided adaptive GGUF quantization with reproducible sweeps.",
    path: "/rl-quant",
    tag: "Research",
    Icon: IconQuant,
  },
  {
    id: "compress",
    title: "Model Compression",
    desc: "Distill → prune → recover → GPTQ/AWQ with lm-eval benchmarks.",
    path: "/compress",
    Icon: IconCompress,
  },
  {
    id: "recipes",
    title: "Recipe Studio",
    desc: "Visual data pipelines for dataset prep and experiment prep.",
    path: "/recipes",
    Icon: IconRecipes,
  },
  {
    id: "knowledge",
    title: "Knowledge Base",
    desc: "Local RAG corpus with on-device chunking and retrieval.",
    path: "/knowledge",
    Icon: IconKnowledge,
  },
] as const;

export function DashboardPage() {
  const { profile: hw } = useHardwareProfile();
  const metrics = useLiveMetrics(true);
  const [goal, setGoal] = useState<string>("chat");
  const [steps, setSteps] = useState<GuideStep[]>([]);

  useEffect(() => {
    api.guide(goal).then((r) => setSteps(r.steps)).catch(console.error);
  }, [goal]);

  const activeGoal = GOALS.find((g) => g.id === goal) ?? GOALS[0];
  const goalPath = useMemo(() => {
    if (activeGoal.id === "chat") {
      return chatPath({ repo: hw?.recommended_chat_repo ?? null });
    }
    return activeGoal.path;
  }, [activeGoal, hw?.recommended_chat_repo]);

  const vramTotal = hw?.gpus[0]?.vram_total_mb;
  const vramUsed = hw?.gpus[0]?.vram_used_mb;
  const headroomGb = hw?.vram_headroom_mb != null
    ? Math.max(0, Math.round(hw.vram_headroom_mb / 1024))
    : null;
  const isAppleUnified = hw?.tier_label?.toLowerCase().includes("unified");

  return (
    <div className="dashboard">
      <PageHeader
        title="Dashboard"
        subtitle="Hardware-aware guidance — everything stays on this machine, nothing is sent elsewhere."
        group="Overview"
        badge={
          <div className="header-badges">
            <span className="trust-badge">
              <span className="live-dot" />
              Local only
            </span>
            {hw?.tier_label && <span className="trust-badge trust-badge-dim">{hw.tier_label}</span>}
          </div>
        }
      />

      {hw && (
        <div className="hw-grid">
          <div className="card hw-card hw-card-accent">
            <div className="hw-card-icon"><IconCpu size={18} /></div>
            <div className="hw-card-label">Processor</div>
            <div className="hw-card-value">{hw.cpu_brand}</div>
            <div className="hw-card-meta">{hw.cpu_cores} cores · {hw.arch}</div>
          </div>
          <div className="card hw-card">
            <div className="hw-card-icon"><IconMemory size={18} /></div>
            <div className="hw-card-label">Memory</div>
            <div className="hw-card-value">{hw.ram_gb} GB RAM</div>
            {headroomGb != null && (
              <div className="hw-card-meta">
                ~{headroomGb} GB free now
                {isAppleUnified && hw.ram_gb ? ` · ${Math.round(hw.ram_gb)} GB unified pool` : ""}
              </div>
            )}
          </div>
          <div className="card hw-card">
            <div className="hw-card-icon"><IconChat size={18} /></div>
            <div className="hw-card-label">Inference backend</div>
            <div className="hw-card-value">{hw.backend.toUpperCase()}</div>
            <div className="hw-card-meta">{hw.platform}</div>
          </div>
          {hw.gpus.length > 0 ? (
            hw.gpus.map((g, i) => (
              <div key={i} className="card hw-card">
                <div className="hw-card-icon"><IconGpu size={18} /></div>
                <div className="hw-card-label">GPU {hw.gpus.length > 1 ? i + 1 : ""}</div>
                <div className="hw-card-value">{g.name}</div>
                <div className="hw-card-meta">
                  {g.vram_total_mb ? `${Math.round(g.vram_total_mb / 1024)} GB VRAM` : "Unified memory"}
                </div>
              </div>
            ))
          ) : (
            <div className="card hw-card">
              <div className="hw-card-icon"><IconGpu size={18} /></div>
              <div className="hw-card-label">GPU</div>
              <div className="hw-card-value">CPU / unified</div>
              <div className="hw-card-meta">Use small models</div>
            </div>
          )}
        </div>
      )}

      {metrics && (
        <div className="card live-metrics-bar">
          <span className="live-dot" />
          <span className="live-metric">CPU <strong>{metrics.cpu_util_pct ?? "—"}%</strong></span>
          <span className="live-metric">RAM <strong>{metrics.ram_used_pct}%</strong></span>
          {metrics.gpus[0] && (
            <>
              <span className="live-metric">GPU <strong>{metrics.gpus[0].utilization_pct ?? "—"}%</strong></span>
              {metrics.gpus[0].temperature_c != null && (
                <span className="live-metric">{metrics.gpus[0].temperature_c}°C</span>
              )}
            </>
          )}
        </div>
      )}

      <section className="goal-section">
        <div className="section-head">
          <h2 className="section-title">What do you want to do?</h2>
          <p className="section-desc">Pick a workflow — recommendations adapt to your hardware.</p>
        </div>
        <div className="goal-grid">
          {GOALS.map((g) => (
            <button
              key={g.id}
              type="button"
              className={`goal-card${goal === g.id ? " goal-card-active" : ""}`}
              onClick={() => setGoal(g.id)}
              aria-pressed={goal === g.id}
            >
              {goal === g.id && <span className="goal-card-check" aria-hidden>✓</span>}
              <span className="goal-icon-wrap">
                <g.Icon size={20} />
              </span>
              <span className="goal-label">{g.label}</span>
              <span className="goal-desc">{g.desc}</span>
            </button>
          ))}
        </div>
      </section>

      <section className="research-section">
        <div className="section-head">
          <h2 className="section-title">Research pipelines</h2>
          <p className="section-desc">Reproducible compression, quantization, and retrieval workflows — all local.</p>
        </div>
        <PipelineStrip pipelines={[...RESEARCH_PIPELINES]} />
      </section>

      {steps.length > 0 && (
        <section className="card guide-section">
          <div className="guide-section-head">
            <div className="section-head">
              <h2 className="section-title">Recommended next steps</h2>
              <p className="section-desc">Tailored for <strong>{activeGoal.label}</strong> on your hardware.</p>
            </div>
            <Link to={goalPath} className="btn btn-primary guide-cta">
              Open {activeGoal.label}
              <IconChevronRight size={14} />
            </Link>
          </div>
          <ol className="guide-list">
            {steps.map((s, i) => (
              <li key={i}>
                <Link to={s.path} className="guide-link">
                  <span className="guide-step-num">{String(i + 1).padStart(2, "0")}</span>
                  <span className="guide-link-body">
                    <strong>{s.title}</strong>
                    <span>{s.detail}</span>
                  </span>
                  <IconChevronRight size={16} className="guide-link-arrow" />
                </Link>
              </li>
            ))}
          </ol>
          {vramTotal != null && vramTotal > 0 ? (
            <p className="muted-text guide-vram">
              VRAM headroom: ~{Math.max(0, Math.round((vramTotal - (vramUsed ?? 0)) / 1024))} GB available
            </p>
          ) : headroomGb != null && isAppleUnified ? (
            <p className="muted-text guide-vram">
              Memory: ~{headroomGb} GB free of {Math.round(hw?.ram_gb ?? 0)} GB unified — close other apps if models look blocked.
            </p>
          ) : null}
        </section>
      )}
    </div>
  );
}
