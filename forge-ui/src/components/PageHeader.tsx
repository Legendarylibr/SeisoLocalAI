type PageHeaderProps = {
  title: string;
  subtitle?: string;
  group?: string;
  badge?: React.ReactNode;
  actions?: React.ReactNode;
  className?: string;
};

export function PageHeader({ title, subtitle, group, badge, actions, className = "" }: PageHeaderProps) {
  return (
    <header className={`page-header${className ? ` ${className}` : ""}`}>
      {group && (
        <nav className="page-breadcrumb" aria-label="Section">
          <span className="page-breadcrumb-group">{group}</span>
          <span className="page-breadcrumb-sep" aria-hidden>/</span>
          <span className="page-breadcrumb-current">{title}</span>
        </nav>
      )}
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
