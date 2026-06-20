import { type ReactNode } from "react";

type StudioCardHeaderProps = {
  icon?: ReactNode;
  title: string;
  description?: string;
  meta?: ReactNode;
  inline?: boolean;
  tone?: "default" | "monitor" | "history";
};

export function StudioCardHeader({
  icon,
  title,
  description,
  meta,
  inline = false,
  tone = "default",
}: StudioCardHeaderProps) {
  return (
    <div
      className={[
        "studio-card-head",
        inline ? "studio-card-head-inline" : "",
        tone !== "default" ? `studio-card-head-${tone}` : "",
      ].filter(Boolean).join(" ")}
      title={description}
    >
      {icon != null && (
        <span className={`studio-card-icon studio-card-icon-${tone}`} aria-hidden>
          {icon}
        </span>
      )}
      <div className="studio-card-head-text">
        <div className="studio-card-title">{title}</div>
        {description && <div className="studio-card-desc">{description}</div>}
      </div>
      {meta && <div className="studio-card-head-aside">{meta}</div>}
    </div>
  );
}
