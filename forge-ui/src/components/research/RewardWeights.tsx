const REWARD_LABELS: Record<string, { label: string; hint: string }> = {
  alpha_latency: { label: "Latency (α)", hint: "Penalize slow inference" },
  beta_throughput: { label: "Throughput (β)", hint: "Reward tokens/sec" },
  gamma_perplexity: { label: "Perplexity (γ)", hint: "Quality preservation weight" },
  delta_memory: { label: "Memory (δ)", hint: "VRAM footprint penalty" },
  epsilon_instability: { label: "Stability (ε)", hint: "Quantization variance penalty" },
  theta_kernel_speedup: { label: "Kernel speedup (θ)", hint: "Reward fused CUDA profile speedup" },
  iota_kernel_latency: { label: "Kernel latency (ι)", hint: "Penalize slow kernel micro-benchmarks" },
};

type RewardWeightsProps = {
  weights: Record<string, number>;
  onChange: (weights: Record<string, number>) => void;
};

export function RewardWeights({ weights, onChange }: RewardWeightsProps) {
  const total = Object.values(weights).reduce((a, b) => a + b, 0) || 1;

  return (
    <div className="reward-weights">
      {Object.entries(weights).map(([key, val]) => {
        const meta = REWARD_LABELS[key] ?? { label: key, hint: "" };
        const pct = (val / total) * 100;
        return (
          <div key={key} className="reward-weight-row">
            <div className="reward-weight-head">
              <span className="reward-weight-label">{meta.label}</span>
              <span className="reward-weight-value">{val.toFixed(3)}</span>
            </div>
            {meta.hint && <span className="reward-weight-hint">{meta.hint}</span>}
            <div className="reward-weight-track">
              <div className="reward-weight-fill" style={{ width: `${Math.min(100, pct)}%` }} />
            </div>
            <input
              type="range"
              className="reward-weight-slider"
              min={0}
              max={2}
              step={0.001}
              value={val}
              onChange={(e) => onChange({ ...weights, [key]: +e.target.value })}
              aria-label={meta.label}
            />
          </div>
        );
      })}
    </div>
  );
}
