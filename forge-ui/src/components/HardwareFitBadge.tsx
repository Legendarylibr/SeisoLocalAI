type Fit = "ideal" | "good" | "tight" | "unlikely" | string;

const FIT_CLASS: Record<string, string> = {
  ideal: "fit-ideal",
  good: "fit-good",
  tight: "fit-tight",
  unlikely: "fit-unlikely",
};

export function HardwareFitBadge({ fit, label }: { fit?: Fit; label?: string }) {
  if (!fit || fit === "unknown") return null;
  const text = label || fit;
  return <span className={`hw-fit-badge ${FIT_CLASS[fit] || ""}`}>{text}</span>;
}
