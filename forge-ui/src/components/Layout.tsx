import "./styles.css";

export function Layout({ children }: { children: React.ReactNode }) {
  return (
    <div className="layout">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-mark">◈</span>
          <span>Seiso Forge</span>
        </div>
        <nav>
          <a href="/">Hub</a>
          <a href="/chat">Chat</a>
          <a href="/train">Train</a>
          <a href="/export">Export</a>
          <a href="/recipes">Recipes</a>
          <a href="/integrations">Integrations</a>
          <a href="/settings">Settings</a>
        </nav>
      </aside>
      <main className="content">{children}</main>
    </div>
  );
}
