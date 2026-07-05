import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { PageHeader } from "@/components/PageHeader";
import { IconIntegrations } from "@/components/Icons";

type Provider = { id: string; name: string; provider_type: string; config: Record<string, unknown> };

export function IntegrationsPage() {
  const [providers, setProviders] = useState<Provider[]>([]);
  const [pName, setPName] = useState("");
  const [pType, setPType] = useState("vllm");
  const [pKey, setPKey] = useState("");
  const [pBaseUrl, setPBaseUrl] = useState("");
  const [pModel, setPModel] = useState("");

  const refresh = async () => {
    setProviders(await api.listProviders());
  };

  useEffect(() => {
    refresh().catch(console.error);
  }, []);

  const addProvider = async () => {
    const config: Record<string, unknown> = {
      api_key: pKey,
      model: pModel || "default",
    };
    if (pBaseUrl.trim()) config.base_url = pBaseUrl.trim();
    await api.createProvider({
      name: pName,
      provider_type: pType,
      config,
    });
    setPName("");
    setPKey("");
    setPBaseUrl("");
    setPModel("");
    refresh();
  };

  return (
    <div>
      <PageHeader
        title="Integrations"
        subtitle="Local inference backends — vLLM."
        group="Platform"
      />

      <div className="card">
        <div className="card-head">
          <span className="card-head-icon" aria-hidden>
            <IconIntegrations size={18} />
          </span>
          <div className="card-head-text">
            <h3>Local backends</h3>
            <p>Connect a local vLLM server. Optional API keys are encrypted at rest.</p>
          </div>
        </div>
        <div className="grid" style={{ marginBottom: "1rem" }}>
          <div>
            <label>Name</label>
            <input value={pName} onChange={(e) => setPName(e.target.value)} placeholder="My vLLM server" />
          </div>
          <div>
            <label>Type</label>
            <select value={pType} onChange={(e) => setPType(e.target.value)}>
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
              placeholder="sk-… (optional for local servers)"
            />
          </div>
          <div>
            <label>Base URL (optional)</label>
            <input
              value={pBaseUrl}
              onChange={(e) => setPBaseUrl(e.target.value)}
              placeholder="http://127.0.0.1:8000/v1"
            />
          </div>
          <div>
            <label>Model (optional)</label>
            <input
              value={pModel}
              onChange={(e) => setPModel(e.target.value)}
              placeholder="default"
            />
          </div>
        </div>
        <button className="btn btn-primary" onClick={addProvider} disabled={!pName || (!pKey && pType !== "vllm")}>
          Add provider
        </button>
        {providers.length === 0 ? (
          <p className="muted-text" style={{ marginTop: "1rem" }}>No providers configured yet.</p>
        ) : (
          <table style={{ marginTop: "1rem" }}>
            <thead>
                  <tr><th>Name</th><th>Type</th><th>Model</th><th>Key</th><th></th></tr>
            </thead>
            <tbody>
              {providers.map((p) => (
                <tr key={p.id}>
                  <td>{p.name}</td>
                  <td><span className="badge">{p.provider_type}</span></td>
                  <td className="muted-text">{String(p.config.model || "—")}</td>
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
