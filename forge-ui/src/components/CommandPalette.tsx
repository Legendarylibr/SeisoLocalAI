import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { NAV_GROUPS } from "@/components/Layout";
import {
  IconSearch,
  IconChevronRight,
  IconRefresh,
  IconLock,
  NavIcon,
  type NavIconName,
} from "@/components/Icons";
import { useAuth } from "@/hooks/useAuth";
import { useToast } from "@/context/ToastContext";

type Command = {
  id: string;
  label: string;
  group: string;
  hint?: string;
  keywords?: string;
  icon?: NavIconName;
  action: "nav" | "run";
  run: () => void;
};

type CommandPaletteProps = {
  open: boolean;
  onClose: () => void;
};

export function CommandPalette({ open, onClose }: CommandPaletteProps) {
  const navigate = useNavigate();
  const { logout } = useAuth();
  const { notify } = useToast();
  const [query, setQuery] = useState("");
  const [active, setActive] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);

  const commands = useMemo<Command[]>(() => {
    const navCommands: Command[] = NAV_GROUPS.flatMap((group) =>
      group.items.map((item) => ({
        id: `nav:${item.to}`,
        label: item.label,
        group: group.label,
        hint: item.desc,
        keywords: `${item.label} ${item.desc ?? ""} ${group.label}`,
        icon: item.icon,
        action: "nav" as const,
        run: () => navigate(item.to),
      })),
    );

    const quickActions: Command[] = [
      {
        id: "action:reload",
        label: "Reload workspace",
        group: "Actions",
        hint: "Refresh the Forge UI",
        keywords: "reload refresh restart workspace",
        action: "run",
        run: () => window.location.reload(),
      },
      {
        id: "action:signout",
        label: "Sign out",
        group: "Actions",
        hint: "End your local session",
        keywords: "sign out logout exit lock session",
        action: "run",
        run: () => {
          void logout();
          notify("Signed out", { tone: "info" });
        },
      },
    ];

    return [...navCommands, ...quickActions];
  }, [navigate, logout, notify]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return commands;
    const terms = q.split(/\s+/);
    return commands.filter((cmd) => {
      const haystack = `${cmd.label} ${cmd.keywords ?? ""}`.toLowerCase();
      return terms.every((term) => haystack.includes(term));
    });
  }, [commands, query]);

  useEffect(() => {
    setActive(0);
  }, [query, open]);

  useEffect(() => {
    if (open) {
      setQuery("");
      const raf = requestAnimationFrame(() => inputRef.current?.focus());
      return () => cancelAnimationFrame(raf);
    }
    return undefined;
  }, [open]);

  const runCommand = useCallback(
    (cmd: Command | undefined) => {
      if (!cmd) return;
      onClose();
      cmd.run();
    },
    [onClose],
  );

  const onKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === "ArrowDown") {
        e.preventDefault();
        setActive((i) => Math.min(i + 1, filtered.length - 1));
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        setActive((i) => Math.max(i - 1, 0));
      } else if (e.key === "Enter") {
        e.preventDefault();
        runCommand(filtered[active]);
      } else if (e.key === "Escape") {
        e.preventDefault();
        onClose();
      }
    },
    [filtered, active, runCommand, onClose],
  );

  useEffect(() => {
    const el = listRef.current?.querySelector<HTMLElement>(`[data-index="${active}"]`);
    el?.scrollIntoView({ block: "nearest" });
  }, [active]);

  if (!open) return null;

  return (
    <div
      className="cmdk-backdrop"
      role="button"
      tabIndex={-1}
      aria-label="Close command palette"
      onClick={onClose}
    >
      <div
        className="cmdk-panel"
        role="dialog"
        aria-modal="true"
        aria-label="Command palette"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="cmdk-input-row">
          <IconSearch size={17} className="cmdk-input-icon" />
          <input
            ref={inputRef}
            className="cmdk-input"
            placeholder="Search pages and actions…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={onKeyDown}
            aria-label="Search commands"
            autoComplete="off"
            spellCheck={false}
          />
          <kbd className="cmdk-esc">Esc</kbd>
        </div>

        <div className="cmdk-list" ref={listRef}>
          {filtered.length === 0 ? (
            <div className="cmdk-empty">No matches for “{query}”</div>
          ) : (
            filtered.map((cmd, index) => (
              <button
                type="button"
                key={cmd.id}
                data-index={index}
                className={`cmdk-item${index === active ? " cmdk-item-active" : ""}`}
                onMouseMove={() => setActive(index)}
                onClick={() => runCommand(cmd)}
              >
                <span className="cmdk-item-icon" aria-hidden>
                  {cmd.icon ? (
                    <NavIcon name={cmd.icon} size={16} />
                  ) : cmd.id === "action:reload" ? (
                    <IconRefresh size={16} />
                  ) : (
                    <IconLock size={16} />
                  )}
                </span>
                <span className="cmdk-item-body">
                  <span className="cmdk-item-label">{cmd.label}</span>
                  {cmd.hint && <span className="cmdk-item-hint">{cmd.hint}</span>}
                </span>
                <span className="cmdk-item-group">{cmd.group}</span>
                <IconChevronRight size={15} className="cmdk-item-arrow" />
              </button>
            ))
          )}
        </div>

        <div className="cmdk-foot">
          <span className="cmdk-foot-hint">
            <kbd>↑</kbd>
            <kbd>↓</kbd>
            to navigate
          </span>
          <span className="cmdk-foot-hint">
            <kbd>↵</kbd>
            to select
          </span>
          <span className="cmdk-foot-hint cmdk-foot-hint-end">Seiso · local only</span>
        </div>
      </div>
    </div>
  );
}
