import { type ReactNode } from "react";

export function StudioCardBody({ children, className = "" }: { children: ReactNode; className?: string }) {
  return <div className={`studio-card-body${className ? ` ${className}` : ""}`}>{children}</div>;
}
