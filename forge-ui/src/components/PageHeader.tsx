type PageHeaderProps = {
  title: string;
  subtitle?: string;
  badge?: React.ReactNode;
  actions?: React.ReactNode;
};

export function PageHeader({ title, subtitle, badge, actions }: PageHeaderProps) {
  return (
    <header className="page-header">
      <div className="page-header-main">
        <div className="page-header-text">
          <h1 className="page-title">{title}</h1>
          {subtitle && <p className="page-sub">{subtitle}</p>}
        </div>
        {(badge || actions) && (
          <div className="page-header-actions">
            {badge}
            {actions}
          </div>
        )}
      </div>
      <div className="page-header-rule" aria-hidden />
    </header>
  );
}
