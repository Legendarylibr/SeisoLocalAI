import type { ReactNode } from "react";

type StatusCalloutProps = {
  tone: "success" | "warn" | "info" | "error";
  title: string;
  children?: ReactNode;
  action?: ReactNode;
};

export function StatusCallout({ tone, title, children, action }: StatusCalloutProps) {
  return (
    <div className={`status-callout status-callout-${tone}`} role="status">
      <div className="status-callout-body">
        <strong className="status-callout-title">{title}</strong>
        {children && <div className="status-callout-text">{children}</div>}
      </div>
      {action && <div className="status-callout-action">{action}</div>}
    </div>
  );
}
