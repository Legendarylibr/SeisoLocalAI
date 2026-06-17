import { PageHeader } from "@/components/PageHeader";

type StudioPageShellProps = {
  title: string;
  subtitle: string;
  group?: string;
  badge?: React.ReactNode;
  banner?: React.ReactNode;
  actions?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
};

export function StudioPageShell({
  title,
  subtitle,
  group = "Studio",
  badge,
  banner,
  actions,
  children,
  className = "",
}: StudioPageShellProps) {
  return (
    <div className={`studio-page${className ? ` ${className}` : ""}`}>
      <PageHeader
        title={title}
        subtitle={subtitle}
        group={group}
        badge={badge}
        actions={actions}
      />
      {banner}
      <div className="studio-page-body">{children}</div>
    </div>
  );
}
