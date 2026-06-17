import { NavLink } from "react-router-dom";
import "../styles.css";

const NAV = [
  { to: "/", label: "Hub", end: true },
  { to: "/chat", label: "Chat" },
  { to: "/train", label: "Train" },
  { to: "/rl-quant", label: "RL Quant" },
  { to: "/compress", label: "Compress" },
  { to: "/image-compress", label: "Image Compress" },
  { to: "/export", label: "Export" },
  { to: "/recipes", label: "Recipes" },
  { to: "/integrations", label: "Integrations" },
  { to: "/settings", label: "Settings" },
];

export function Layout({ children }: { children: React.ReactNode }) {
  return (
    <div className="layout">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-mark">◈</span>
          <span>Seiso Forge</span>
        </div>
        <nav>
          {NAV.map(({ to, label, end }) => (
            <NavLink
              key={to}
              to={to}
              end={end}
              className={({ isActive }) => (isActive ? "active-nav" : undefined)}
            >
              {label}
            </NavLink>
          ))}
        </nav>
        <div className="sidebar-foot">
          <NavLink to="/settings" className="security-link">
            <span aria-hidden>◈</span> Security
          </NavLink>
        </div>
      </aside>
      <main className="content">{children}</main>
    </div>
  );
}
