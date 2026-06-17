import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, SecurityPosture } from "@/lib/api";
import { PageHeader } from "@/components/PageHeader";
import { Tabs } from "@/components/Tabs";
import { IconAlert, IconIntegrations, IconPlug } from "@/components/Icons";

type Provider = { id: string; name: string; provider_type: string; config: Record<string, unknown> };
type McpServer = { id: string; name: string; command: string; args: string[]; enabled: boolean };
type IntegrationsTab = "providers" | "mcp";

export function IntegrationsPage() {
  const [tab, setTab] = useState<IntegrationsTab>("providers");
  const [providers, setProviders] = useState<Provider[]>([]);
  const [servers, setServers] = useState<McpServer[]>([]);
  const [security, setSecurity] = useState<SecurityPosture | null>(null);
  const [pName, setPName] = useState("");
  const [pType, setPType] = useState("openai");
  const [pKey, setPKey] = useState("");
  const [mName, setMName] = useState("");
  const [mCmd, setMCmd] = useState("npx");
  const [mArgs, setMArgs] = useState("@modelcontextprotocol/server-everything@0.6.2");

  const refresh = async () => {
    setProviders(await api.listProviders());
    setServers(await api.listMcpServers());
  };

  useEffect(() => {
    refresh().catch(console.error);
    api.settings().then((s) => setSecurity(s.security)).catch(() => {});
  }, []);

  const addProvider = async () => {
    await api.createProvider({
      name: pName,
      provider_type: pType,
      config: { api_key: pKey, model: pType === "anthropic" ? "claude-3-5-sonnet-20241022" : "gpt-4o-mini" },
    });
    setPName("");
    setPKey("");
    refresh();
  };

  const addMcp = async () => {
    await api.createMcpServer({ name: mName, command: mCmd, args: mArgs.split(" ").filter(Boolean) });
    setMName("");
    refresh();
  };

  return (
    <div>
      <PageHeader
        title="Integrations"
        subtitle="External LLM providers and MCP servers — keys encrypted at rest."
        group="Platform"
      />

      {security && !security.allow_tools && tab === "mcp" && (
        <div className="security-banner">
          <span className="security-banner-icon" aria-hidden>
            <IconAlert size={14} />
          </span>
          <div>
            <strong>MCP requires tools to be enabled</strong>
            <p>
              Set <code>SEISO_ALLOW_TOOLS=true</code> and restart the server before connecting MCP servers in Chat.
              Provider routing works without this flag.
            </p>
          </div>
        </div>
      )}

      <Tabs
        className="integrations-tab-bar"
        aria-label="Integration types"
        value={tab}
        onChange={setTab}
        items={[
          {
            id: "providers",
            label: "LLM providers",
            description: "OpenAI, Anthropic, Ollama, vLLM routing",
            icon: <IconIntegrations size={16} />,
            count: providers.length,
          },
          {
            id: "mcp",
            label: "MCP servers",
            description: "Model Context Protocol tools for Chat",
            icon: <IconPlug size={16} />,
            count: servers.length,
          },
        ]}
      />

      {tab === "providers" && (
        <div className="tab-panel">
          <div className="card">
            <div className="card-head">
              <span className="card-head-icon" aria-hidden>
                <IconIntegrations size={18} />
              </span>
              <div className="card-head-text">
                <h3>External providers</h3>
                <p>API keys are encrypted in the local database and never returned in full after saving.</p>
              </div>
            </div>
            <div className="grid" style={{ marginBottom: "1rem" }}>
              <div>
                <label>Name</label>
                <input value={pName} onChange={(e) => setPName(e.target.value)} placeholder="My OpenAI" />
              </div>
              <div>
                <label>Type</label>
                <select value={pType} onChange={(e) => setPType(e.target.value)}>
                  <option value="openai">OpenAI</option>
                  <option value="anthropic">Anthropic</option>
                  <option value="ollama">Ollama</option>
                  <option value="vllm">vLLM</option>
                </select>
              </div>
              <div>
                <label>API key</label>
                <input
                  type="password"
                  value={pKey}
                  onChange={(e) => setPKey(e.target.value)}
                  autoComplete="off"
                  placeholder="sk-…"
                />
              </div>
            </div>
            <button className="btn btn-primary" onClick={addProvider} disabled={!pName || !pKey}>
              Add provider
            </button>
            {providers.length === 0 ? (
              <p className="muted-text" style={{ marginTop: "1rem" }}>No providers configured yet.</p>
            ) : (
              <table style={{ marginTop: "1rem" }}>
                <thead>
                  <tr><th>Name</th><th>Type</th><th>Key</th><th></th></tr>
                </thead>
                <tbody>
                  {providers.map((p) => (
                    <tr key={p.id}>
                      <td>{p.name}</td>
                      <td><span className="badge">{p.provider_type}</span></td>
                      <td className="muted-text">Key: {String(p.config.api_key || "—")}</td>
                      <td>
                        <button className="btn" onClick={() => api.deleteProvider(p.id).then(refresh)}>Remove</button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </div>
      )}

      {tab === "mcp" && (
        <div className="tab-panel">
          <div className="card">
            <div className="card-head">
              <span className="card-head-icon" aria-hidden>
                <IconPlug size={18} />
              </span>
              <div className="card-head-text">
                <h3>MCP servers</h3>
                <p>Only pinned package versions are allowed. Commands are validated and run with a hardened environment.</p>
              </div>
            </div>
            <label>Name</label>
            <input value={mName} onChange={(e) => setMName(e.target.value)} placeholder="Filesystem MCP" />
            <label>Command</label>
            <input value={mCmd} onChange={(e) => setMCmd(e.target.value)} />
            <label>Args (space-separated, include @version)</label>
            <input value={mArgs} onChange={(e) => setMArgs(e.target.value)} className="mono" />
            <div className="form-actions">
              <button className="btn btn-primary" onClick={addMcp} disabled={!mName}>
                Add MCP server
              </button>
            </div>
            {servers.length === 0 ? (
              <p className="muted-text" style={{ marginTop: "1rem" }}>No MCP servers configured yet.</p>
            ) : (
              <table style={{ marginTop: "1rem" }}>
                <thead>
                  <tr><th>Name</th><th>Command</th><th></th></tr>
                </thead>
                <tbody>
                  {servers.map((s) => (
                    <tr key={s.id}>
                      <td>{s.name}</td>
                      <td className="mono">{s.command} {s.args.join(" ")}</td>
                      <td>
                        <button className="btn" onClick={() => api.connectMcp(s.id).then(refresh)}>Connect</button>
                        {" "}
                        <button className="btn" onClick={() => api.deleteMcpServer(s.id).then(refresh)}>Remove</button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
            <p className="muted-text" style={{ marginTop: "0.75rem", fontSize: "0.8rem" }}>
              Select MCP servers per chat in the <Link to="/chat">Chat</Link> agent tools panel.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
