import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { PageHeader } from "@/components/PageHeader";
import { IconIntegrations } from "@/components/Icons";

type Provider = { id: string; name: string; provider_type: string; config: Record<string, unknown> };

export function IntegrationsPage() {
  const [providers, setProviders] = useState<Provider[]>([]);
  const [pName, setPName] = useState("");
  const [pType, setPType] = useState("openai");
  const [pKey, setPKey] = useState("");

  const refresh = async () => {
    setProviders(await api.listProviders());
  };

  useEffect(() => {
    refresh().catch(console.error);
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

  return (
    <div>
      <PageHeader
        title="Integrations"
        subtitle="External LLM providers — keys encrypted at rest."
        group="Platform"
      />

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
  );
}
