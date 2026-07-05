import { useEffect, useState } from "react";
import { NavLink, useLocation } from "react-router-dom";
import { SystemMonitor } from "@/components/SystemMonitor";
import { CommandPalette } from "@/components/CommandPalette";
import { IconMenu, IconSearch, NavIcon, type NavIconName } from "@/components/Icons";
import { SeisoLogoMark } from "@/components/SeisoLogo";
import { useAuth } from "@/hooks/useAuth";
import { useToast } from "@/context/ToastContext";
import "../styles.css";

const isMac =
  typeof navigator !== "undefined" && /mac|iphone|ipad/i.test(navigator.platform || navigator.userAgent);

export type NavItem = { to: string; label: string; icon: NavIconName; desc?: string; end?: boolean };

export const NAV_GROUPS: { label: string; items: NavItem[] }[] = [
  {
    label: "Overview",
    items: [{ to: "/", label: "Dashboard", icon: "dashboard", desc: "Hardware & workflows", end: true }],
  },
  {
    label: "Models",
    items: [
      { to: "/hub", label: "Hub", icon: "hub", desc: "Browse & download" },
      { to: "/chat", label: "Chat", icon: "chat", desc: "Local inference" },
      { to: "/knowledge", label: "Knowledge", icon: "knowledge", desc: "Local RAG corpus" },
    ],
  },
  {
    label: "Studio",
    items: [
      { to: "/train", label: "Train", icon: "train", desc: "LoRA fine-tuning" },
      { to: "/rl-quant", label: "RL Quant", icon: "quant", desc: "Reward-guided quant" },
      { to: "/compress", label: "Compress", icon: "compress", desc: "Distill & prune LLMs" },
      { to: "/distill-rl", label: "Distill-RL", icon: "recipes", desc: "Distill + DPO alignment" },
      { to: "/export", label: "Export", icon: "export", desc: "Publish to Hub" },
      { to: "/recipes", label: "Recipes", icon: "recipes", desc: "Visual pipelines" },
    ],
  },
  {
    label: "Platform",
    items: [
      { to: "/integrations", label: "Integrations", icon: "integrations", desc: "External LLM providers" },
      { to: "/settings", label: "Settings", icon: "settings", desc: "Account, tokens & security" },
    ],
  },
];

export function Layout({ children, fullBleed = false }: { children: React.ReactNode; fullBleed?: boolean }) {
  const location = useLocation();
  const { logout } = useAuth();
  const { notify } = useToast();
  const isChat = location.pathname === "/chat";
  const isStudioCompact = /^\/(train|rl-quant|compress|distill-rl|export|recipes)(\/|$)/.test(location.pathname);
  const [navOpen, setNavOpen] = useState(false);
  const [paletteOpen, setPaletteOpen] = useState(false);

  useEffect(() => {
    setNavOpen(false);
  }, [location.pathname]);

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setPaletteOpen((v) => !v);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  return (
    <div className={`layout${fullBleed ? " layout-fullbleed" : ""}`}>
      <button
        type="button"
        className="nav-toggle"
        onClick={() => setNavOpen((v) => !v)}
        aria-label="Toggle navigation"
      >
        <IconMenu size={18} />
      </button>

      <button
        type="button"
        className={`sidebar-backdrop${navOpen ? " visible" : ""}`}
        onClick={() => setNavOpen(false)}
        aria-label="Close navigation"
      />

      <aside className={`sidebar${navOpen ? " sidebar-open" : ""}`}>
        <div className="sidebar-mascot" aria-hidden>
          <SeisoLogoMark className="sidebar-mascot-img" />
        </div>

        <NavLink to="/" className="brand sidebar-elevated" style={{ textDecoration: "none", color: "inherit" }}>
          <span className="brand-mark">
            <SeisoLogoMark className="brand-logo-img" />
          </span>
          <span className="brand-text">
            <span className="brand-name">Seiso</span>
            <span className="brand-tagline">Local AI platform</span>
          </span>
        </NavLink>

        <button
          type="button"
          className="sidebar-search sidebar-elevated"
          onClick={() => setPaletteOpen(true)}
          aria-label="Open command palette"
        >
          <IconSearch size={15} className="sidebar-search-icon" />
          <span className="sidebar-search-text">Search…</span>
          <kbd className="sidebar-search-kbd">{isMac ? "⌘" : "Ctrl"} K</kbd>
        </button>

        <div className="sidebar-nav sidebar-elevated">
          {NAV_GROUPS.map((group) => (
            <div key={group.label} className="nav-group">
              <div className="nav-group-label">{group.label}</div>
              <nav>
                {group.items.map(({ to, label, icon, desc, end }) => (
                  <NavLink
                    key={to}
                    to={to}
                    end={end}
                    className={({ isActive }) => `nav-link${isActive ? " active-nav" : ""}`}
                  >
                    <span className="nav-link-icon">
                      <NavIcon name={icon} size={16} />
                    </span>
                    <span className="nav-link-text">
                      <span className="nav-link-label">{label}</span>
                      {desc && <span className="nav-link-desc">{desc}</span>}
                    </span>
                  </NavLink>
                ))}
              </nav>
            </div>
          ))}
        </div>

        <div className="sidebar-foot sidebar-elevated">
          <button
            type="button"
            className="sidebar-foot-link sidebar-foot-btn"
            onClick={() => {
              notify("Signing out…", { tone: "info" });
              void logout();
            }}
          >
            Sign out
          </button>
        </div>
      </aside>

      <main className={`content${isChat ? " content-chat" : ""}${isStudioCompact ? " content-studio-compact" : ""}`}>{children}</main>
      <SystemMonitor />
      <CommandPalette open={paletteOpen} onClose={() => setPaletteOpen(false)} />
    </div>
  );
}
