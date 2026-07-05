import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { StudioPageShell } from "@/components/StudioPageShell";
import { FormSection } from "@/components/research/FormSection";
import { DataTable } from "@/components/research/DataTable";

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
    <StudioPageShell
      title="Integrations"
      subtitle="Connect local vLLM servers and other inference backends for chat routing."
      group="Platform"
    >
      <div className="card studio-card">
        <FormSection
          title="Add provider"
          hint="Optional API keys are encrypted at rest. Local vLLM servers often need no key."
        >
          <div className="grid integrations-form-grid">
            <div className="form-field">
              <label htmlFor="provider-name">Name</label>
              <input
                id="provider-name"
                value={pName}
                onChange={(e) => setPName(e.target.value)}
                placeholder="My vLLM server"
              />
            </div>
            <div className="form-field">
              <label htmlFor="provider-type">Type</label>
              <select id="provider-type" value={pType} onChange={(e) => setPType(e.target.value)}>
                <option value="vllm">vLLM</option>
              </select>
            </div>
            <div className="form-field">
              <label htmlFor="provider-key">API key</label>
              <input
                id="provider-key"
                type="password"
                value={pKey}
                onChange={(e) => setPKey(e.target.value)}
                autoComplete="off"
                placeholder="sk-… (optional for local servers)"
              />
            </div>
            <div className="form-field">
              <label htmlFor="provider-url">Base URL</label>
              <input
                id="provider-url"
                value={pBaseUrl}
                onChange={(e) => setPBaseUrl(e.target.value)}
                placeholder="http://127.0.0.1:8000/v1"
              />
            </div>
            <div className="form-field">
              <label htmlFor="provider-model">Model</label>
              <input
                id="provider-model"
                value={pModel}
                onChange={(e) => setPModel(e.target.value)}
                placeholder="default"
              />
            </div>
          </div>
          <div className="form-actions">
            <button className="btn btn-primary" onClick={addProvider} disabled={!pName || (!pKey && pType !== "vllm")}>
              Add provider
            </button>
          </div>
        </FormSection>

        <FormSection title="Configured providers" hint="Providers appear in Chat under Session settings.">
          <DataTable
            columns={[
              { key: "name", header: "Name" },
              {
                key: "provider_type",
                header: "Type",
                render: (p) => <span className="badge">{p.provider_type}</span>,
              },
              {
                key: "model",
                header: "Model",
                render: (p) => <span className="muted-text">{String(p.config.model || "—")}</span>,
              },
              {
                key: "api_key",
                header: "Key",
                render: (p) => <span className="muted-text">{String(p.config.api_key || "—")}</span>,
                mono: true,
              },
              {
                key: "actions",
                header: "",
                render: (p) => (
                  <button className="btn btn-sm" onClick={() => api.deleteProvider(p.id).then(refresh)}>
                    Remove
                  </button>
                ),
              },
            ]}
            rows={providers}
            getRowKey={(p) => p.id}
            emptyMessage="No providers configured yet. Add a vLLM server above to route chat through it."
          />
        </FormSection>
      </div>
    </StudioPageShell>
  );
}
