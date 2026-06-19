import { SecurityPosture } from "@/lib/api";
import { IconShield } from "@/components/Icons";

type Item = {
  label: string;
  ok: boolean;
  detail: string;
  warn?: boolean;
};

function postureItems(s: SecurityPosture): Item[] {
  return [
    {
      label: "Localhost binding",
      ok: s.bind_localhost,
      detail: s.bind_localhost ? "Server listens on 127.0.0.1 only" : "Remote access enabled — use TLS reverse proxy",
      warn: !s.bind_localhost,
    },
    {
      label: "Encrypted storage",
      ok: s.db_encrypted,
      detail: "Chat and provider keys encrypted at rest (AES-256-GCM)",
    },
    {
      label: "HttpOnly session",
      ok: true,
      detail: `Cookie-based session (${s.session_hours}h) — no tokens in browser storage`,
    },
    {
      label: "Agent tools",
      ok: !s.allow_tools,
      detail: s.allow_tools
        ? "Web search and artifacts are enabled"
        : "Disabled by default (SEISO_ALLOW_TOOLS=false)",
      warn: s.allow_tools,
    },
    {
      label: "Code execution",
      ok: !s.allow_code_exec,
      detail: s.allow_code_exec
        ? "Sandboxed Python enabled — high risk"
        : "Disabled by default (SEISO_ALLOW_CODE_EXEC=false)",
      warn: s.allow_code_exec,
    },
    {
      label: "Rate limiting",
      ok: !s.rate_limit_enabled || s.rate_limit > 0,
      detail: s.rate_limit_enabled
        ? `${s.rate_limit} requests/min per IP`
        : "Disabled for localhost-only mode",
    },
  ];
}

export function SecurityShield({ security }: { security: SecurityPosture }) {
  const items = postureItems(security);
  const score = items.filter((i) => i.ok && !i.warn).length;
  const total = items.length;

  return (
    <div className="security-shield">
      <div className="security-shield-header">
        <span className="security-shield-icon" aria-hidden>
          <IconShield size={22} />
        </span>
        <div>
          <div className="security-shield-title">Security posture</div>
          <div className="security-shield-score">
            {score}/{total} protections active
          </div>
        </div>
      </div>
      <ul className="security-shield-list">
        {items.map((item) => (
          <li key={item.label} className="security-shield-item">
            <span
              className={`security-dot${item.warn ? " security-dot-warn" : item.ok ? " security-dot-ok" : " security-dot-off"}`}
              aria-hidden
            />
            <div>
              <div className="security-item-label">{item.label}</div>
              <div className="security-item-detail">{item.detail}</div>
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}
