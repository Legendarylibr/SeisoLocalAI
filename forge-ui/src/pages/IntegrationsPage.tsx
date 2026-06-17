import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, SecurityPosture } from "@/lib/api";

type Provider = { id: string; name: string; provider_type: string; config: Record<string, unknown> };
type McpServer = { id: string; name: string; command: string; args: string[]; enabled: boolean };

export function IntegrationsPage() {
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
      <h1 className="page-title">Integrations</h1>
      <p className="page-sub">External LLM providers and MCP servers — keys encrypted at rest.</p>

      {security && !security.allow_tools && (
        <div className="security-banner">
          <span className="security-banner-icon" aria-hidden>!</span>
          <div>
            <strong>MCP requires tools to be enabled</strong>
            <p>
              Set <code>SEISO_ALLOW_TOOLS=true</code> and restart the server before connecting MCP servers in Chat.
              Provider routing works without this flag.
            </p>
          </div>
        </div>
      )}

      <div className="card">
        <h3 className="section-title">Providers</h3>
        <p className="muted-text" style={{ marginBottom: "1rem" }}>
          API keys are encrypted in the local database and never returned in full after saving.
        </p>
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
        <table style={{ marginTop: "1rem" }}>
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
      </div>

      <div className="card">
        <h3 className="section-title">MCP servers</h3>
        <p className="muted-text" style={{ marginBottom: "1rem" }}>
          Only pinned package versions are allowed. Commands are validated and run with a hardened environment.
        </p>
        <label>Name</label>
        <input value={mName} onChange={(e) => setMName(e.target.value)} placeholder="Filesystem MCP" />
        <label>Command</label>
        <input value={mCmd} onChange={(e) => setMCmd(e.target.value)} />
        <label>Args (space-separated, include @version)</label>
        <input value={mArgs} onChange={(e) => setMArgs(e.target.value)} className="mono" />
        <button className="btn btn-primary" style={{ marginTop: "0.5rem" }} onClick={addMcp} disabled={!mName}>
          Add MCP server
        </button>
        <table style={{ marginTop: "1rem" }}>
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
        <p className="muted-text" style={{ marginTop: "0.75rem", fontSize: "0.8rem" }}>
          Select MCP servers per chat in the <Link to="/chat">Chat</Link> agent tools panel.
        </p>
      </div>
    </div>
  );
}
