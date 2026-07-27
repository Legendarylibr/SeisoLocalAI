import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import type { ManagedVllmStatus, ProviderRow } from "@/lib/api/providers";
import { PageHeader } from "@/components/PageHeader";
import { IconIntegrations } from "@/components/Icons";

type NostrStatus = {
  server_allow_nostr: boolean;
  key_saved: boolean;
  npub: string | null;
  auto_attest: boolean;
  relays: string[];
  allow_loopback: boolean;
};

export function IntegrationsPage() {
  const [providers, setProviders] = useState<ProviderRow[]>([]);
  const [pName, setPName] = useState("");
  const [pType, setPType] = useState("local_chat");
  const [pKey, setPKey] = useState("");
  const [pBaseUrl, setPBaseUrl] = useState("");
  const [pModel, setPModel] = useState("");
  const [pTp, setPTp] = useState("");
  const [pGpuCount, setPGpuCount] = useState("");
  const [pHoster, setPHoster] = useState("custom");
  const [error, setError] = useState<string | null>(null);
  const [managed, setManaged] = useState<ManagedVllmStatus | null>(null);
  const [mvModel, setMvModel] = useState("");
  const [mvTp, setMvTp] = useState("");
  const [mvBusy, setMvBusy] = useState(false);
  const [compatHint, setCompatHint] = useState<string | null>(null);
  const [nostr, setNostr] = useState<NostrStatus | null>(null);
  const [nostrRelays, setNostrRelays] = useState("");
  const [nostrAuto, setNostrAuto] = useState(false);
  const [nostrLoopback, setNostrLoopback] = useState(false);
  const [nostrImport, setNostrImport] = useState("");
  const [nostrMsg, setNostrMsg] = useState<string | null>(null);

  const refresh = async () => {
    const [plist, mstatus, nstatus] = await Promise.all([
      api.listProviders(),
      api.managedVllmStatus().catch(() => null),
      api.nostrStatus().catch(() => null),
    ]);
    setProviders(plist);
    if (mstatus) {
      setManaged(mstatus);
      if (!mvTp && mstatus.suggested_tensor_parallel) {
        setMvTp(String(mstatus.suggested_tensor_parallel));
      }
    }
    if (nstatus) {
      setNostr(nstatus);
      setNostrRelays((nstatus.relays || []).join(", "));
      setNostrAuto(Boolean(nstatus.auto_attest));
      setNostrLoopback(Boolean(nstatus.allow_loopback));
    }
  };

  useEffect(() => {
    refresh().catch(console.error);
  }, []);

  const isRemote = pType === "remote_chat";

  const addProvider = async () => {
    setError(null);
    setCompatHint(null);
    try {
      const config: Record<string, unknown> = {
        model: pModel || "default",
      };
      if (pKey) config.api_key = pKey;
      if (pBaseUrl.trim()) config.base_url = pBaseUrl.trim();
      if (isRemote) {
        config.deployment_kind = "multi_gpu_remote";
        if (pTp) config.tensor_parallel_size = Number(pTp);
        if (pGpuCount) config.gpu_count = Number(pGpuCount);
        if (pHoster) config.hoster = pHoster;
      } else if (pTp) {
        config.tensor_parallel_size = Number(pTp);
        config.deployment_kind = "multi_gpu_local";
      }
      const created = await api.createProvider({
        name: pName,
        provider_type: pType,
        config,
      });
      setPName("");
      setPKey("");
      setPBaseUrl("");
      setPModel("");
      setPTp("");
      setPGpuCount("");
      setCompatHint(
        `External agents: GET /v1/models · chat with model "provider:${created.id}"` +
          (pModel ? ` or "${pModel}"` : "") +
          ` at http://127.0.0.1:8765/v1`
      );
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const startManaged = async () => {
    setError(null);
    setCompatHint(null);
    setMvBusy(true);
    try {
      const body: Record<string, unknown> = { model: mvModel.trim() };
      if (mvTp) body.tensor_parallel_size = Number(mvTp);
      const res = await api.managedVllmStart(body);
      const ids = res.compat?.model_ids?.join(", ") || "";
      setCompatHint(
        res.compat?.note
          ? `${res.compat.note} Model ids: ${ids}`
          : `Compat API model ids: ${ids}`
      );
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setMvBusy(false);
    }
  };

  const stopManaged = async () => {
    setError(null);
    setMvBusy(true);
    try {
      await api.managedVllmStop();
      setCompatHint(null);
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setMvBusy(false);
    }
  };

  const cloudEnabled = Boolean(managed?.cloud_multigpu_enabled);
  const managedEnabled = Boolean(managed?.feature_enabled ?? managed?.enabled);
  const needsKey = isRemote;
  const canAdd =
    Boolean(pName.trim()) &&
    (!isRemote || (Boolean(pBaseUrl.trim()) && Boolean(pModel.trim()))) &&
    (!needsKey || Boolean(pKey));

  return (
    <div>
      <PageHeader
        title="Integrations"
        subtitle="Local and remote multi-GPU chat servers. Vendor-neutral — any chat-completions HTTP endpoint. External agents use Forge Compat API /v1."
        group="Platform"
      />

      {error && (
        <p className="muted-text" style={{ color: "var(--danger, #f66)", marginBottom: "1rem" }}>
          {error}
        </p>
      )}
      {compatHint && (
        <p className="muted-text" style={{ marginBottom: "1rem" }}>
          {compatHint}
        </p>
      )}

      <div className="card" style={{ marginBottom: "1rem" }}>
        <div className="card-head">
          <span className="card-head-icon" aria-hidden>
            <IconIntegrations size={18} />
          </span>
          <div className="card-head-text">
            <h3>Connect chat server</h3>
            <p>
              Point Chat and the Compat API at any standard chat-completions endpoint
              (vLLM, SGLang, hosted pods, etc.). Local uses loopback; remote multi-GPU
              requires <code>SEISO_ALLOW_CLOUD_MULTIGPU=true</code>.
            </p>
          </div>
        </div>
        <div className="grid" style={{ marginBottom: "1rem" }}>
          <div>
            <label>Name</label>
            <input value={pName} onChange={(e) => setPName(e.target.value)} placeholder="My multi-GPU server" />
          </div>
          <div>
            <label>Type</label>
            <select value={pType} onChange={(e) => setPType(e.target.value)}>
              <option value="local_chat">Local chat server (loopback)</option>
              <option value="remote_chat" disabled={!cloudEnabled}>
                Remote chat server (HTTPS multi-GPU)
                {cloudEnabled ? "" : " — enable SEISO_ALLOW_CLOUD_MULTIGPU"}
              </option>
            </select>
          </div>
          <div>
            <label>API key</label>
            <input
              type="password"
              value={pKey}
              onChange={(e) => setPKey(e.target.value)}
              autoComplete="off"
              placeholder={needsKey ? "required for remote" : "optional for local"}
            />
          </div>
          <div>
            <label>Base URL</label>
            <input
              value={pBaseUrl}
              onChange={(e) => setPBaseUrl(e.target.value)}
              placeholder={
                isRemote
                  ? "https://your-pod.example.com/v1"
                  : "http://127.0.0.1:8000/v1"
              }
            />
          </div>
          <div>
            <label>Model</label>
            <input
              value={pModel}
              onChange={(e) => setPModel(e.target.value)}
              placeholder={isRemote ? "served model id" : "default"}
            />
          </div>
          <div>
            <label>Tensor parallel (optional metadata)</label>
            <input value={pTp} onChange={(e) => setPTp(e.target.value)} placeholder="e.g. 4" />
          </div>
          {isRemote && (
            <>
              <div>
                <label>GPU count (optional metadata)</label>
                <input value={pGpuCount} onChange={(e) => setPGpuCount(e.target.value)} placeholder="e.g. 8" />
              </div>
              <div>
                <label>Hoster (optional metadata)</label>
                <select value={pHoster} onChange={(e) => setPHoster(e.target.value)}>
                  <option value="custom">custom</option>
                  <option value="runpod">runpod</option>
                  <option value="lambda">lambda</option>
                  <option value="coreweave">coreweave</option>
                  <option value="aws">aws</option>
                  <option value="gcp">gcp</option>
                  <option value="azure">azure</option>
                </select>
              </div>
            </>
          )}
        </div>
        <button className="btn btn-primary" onClick={() => void addProvider()} disabled={!canAdd}>
          Add chat server
        </button>
        {providers.length === 0 ? (
          <p className="muted-text" style={{ marginTop: "1rem" }}>No chat servers configured yet.</p>
        ) : (
          <table style={{ marginTop: "1rem" }}>
            <thead>
              <tr>
                <th>Name</th>
                <th>Type</th>
                <th>Model</th>
                <th>Compat model id</th>
                <th>Key</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {providers.map((p) => (
                <tr key={p.id}>
                  <td>{p.name}</td>
                  <td><span className="badge">{p.provider_type}</span></td>
                  <td className="muted-text">{String(p.config.model || "—")}</td>
                  <td className="muted-text"><code>provider:{p.id}</code></td>
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

      <div className="card">
        <div className="card-head">
          <span className="card-head-icon" aria-hidden>
            <IconIntegrations size={18} />
          </span>
          <div className="card-head-text">
            <h3>Managed local multi-GPU (optional)</h3>
            <p>
              Optionally start a local multi-GPU chat server (engine: vLLM). Off by default —
              set <code>SEISO_MANAGED_VLLM_ENABLED=true</code>. Free memory stops it.
              Registers a <code>local_chat</code> server for Chat and external agents via{" "}
              <code>/v1</code>.
            </p>
          </div>
        </div>
        {!managedEnabled ? (
          <p className="muted-text">
            Feature disabled. Enable with <code>SEISO_MANAGED_VLLM_ENABLED=true</code> in{" "}
            <code>.env</code>, install <code>vllm</code>, then restart Forge.
          </p>
        ) : (
          <>
            <p className="muted-text" style={{ marginBottom: "0.75rem" }}>
              Status:{" "}
              {managed?.running
                ? `running${managed.healthy ? " (healthy)" : ""} · TP ${managed.tensor_parallel_size ?? "?"} · ${managed.model || ""}`
                : "stopped"}
              {typeof managed?.gpu_count === "number" ? ` · ${managed.gpu_count} GPU(s) detected` : ""}
              {managed?.vllm_available === false ? " · vLLM not installed in this environment" : ""}
            </p>
            <div className="grid" style={{ marginBottom: "1rem" }}>
              <div>
                <label>Model (HF id or path)</label>
                <input
                  value={mvModel}
                  onChange={(e) => setMvModel(e.target.value)}
                  placeholder="org/model-id"
                />
              </div>
              <div>
                <label>Tensor parallel</label>
                <input
                  value={mvTp}
                  onChange={(e) => setMvTp(e.target.value)}
                  placeholder={String(managed?.suggested_tensor_parallel || 1)}
                />
              </div>
            </div>
            <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
              <button
                className="btn btn-primary"
                disabled={mvBusy || !mvModel.trim() || managed?.running}
                onClick={() => void startManaged()}
              >
                {mvBusy ? "Working…" : "Start managed multi-GPU"}
              </button>
              <button
                className="btn"
                disabled={mvBusy || !managed?.running}
                onClick={() => void stopManaged()}
              >
                Stop
              </button>
            </div>
          </>
        )}
      </div>

      <div className="card" style={{ marginTop: "1rem" }}>
        <div className="card-head">
          <span className="card-head-icon" aria-hidden>
            <IconIntegrations size={18} />
          </span>
          <div className="card-head-text">
            <h3>Nostr provenance</h3>
            <p>
              External attestation of run manifest digests (not weights). Default path is on;
              disable with <code>SEISO_ALLOW_NOSTR=0</code>. Optional deps:{" "}
              <code>pip install &apos;seiso[nostr]&apos;</code>.
            </p>
          </div>
        </div>
        {nostr && !nostr.server_allow_nostr && (
          <p className="muted-text" style={{ marginBottom: "0.75rem" }}>
            Server gate off — remove <code>SEISO_ALLOW_NOSTR=0</code> (or set{" "}
            <code>SEISO_ALLOW_NOSTR=1</code>) and restart Forge to publish.
          </p>
        )}
        <table className="status-table" style={{ marginBottom: "0.75rem" }}>
          <tbody>
            <tr>
              <td>Key saved</td>
              <td>{nostr?.key_saved ? "Yes" : "No"}</td>
            </tr>
            <tr>
              <td>npub</td>
              <td className="mono">{nostr?.npub || "—"}</td>
            </tr>
          </tbody>
        </table>
        <label>Allowlisted relays (comma-separated wss://)</label>
        <input
          value={nostrRelays}
          onChange={(e) => setNostrRelays(e.target.value)}
          placeholder="wss://relay.example.com"
        />
        <label style={{ display: "flex", gap: "0.5rem", alignItems: "center", marginTop: "0.75rem" }}>
          <input
            type="checkbox"
            checked={nostrAuto}
            onChange={(e) => setNostrAuto(e.target.checked)}
          />
          Auto-attest completed pipeline / export / RL-quant jobs
        </label>
        <label style={{ display: "flex", gap: "0.5rem", alignItems: "center", marginTop: "0.5rem" }}>
          <input
            type="checkbox"
            checked={nostrLoopback}
            onChange={(e) => setNostrLoopback(e.target.checked)}
          />
          Allow loopback ws:// relays (dev only)
        </label>
        <div className="form-actions" style={{ marginTop: "0.75rem" }}>
          <button
            className="btn btn-primary"
            onClick={() => {
              setNostrMsg(null);
              const relays = nostrRelays
                .split(",")
                .map((r) => r.trim())
                .filter(Boolean);
              void api
                .saveNostrPrefs({
                  auto_attest: nostrAuto,
                  relays,
                  allow_loopback: nostrLoopback,
                })
                .then(() => {
                  setNostrMsg("Preferences saved");
                  return refresh();
                })
                .catch((e) => setNostrMsg(e instanceof Error ? e.message : String(e)));
            }}
          >
            Save Nostr prefs
          </button>
          <button
            className="btn"
            onClick={() => {
              setNostrMsg(null);
              void api
                .nostrKeygen()
                .then((res) => {
                  setNostrMsg(`Key created · ${res.npub}`);
                  return refresh();
                })
                .catch((e) => setNostrMsg(e instanceof Error ? e.message : String(e)));
            }}
          >
            Generate key
          </button>
          {nostr?.key_saved && (
            <button
              className="btn"
              onClick={() => {
                setNostrMsg(null);
                void api
                  .clearNostrKey()
                  .then(() => {
                    setNostrMsg("Key cleared");
                    return refresh();
                  })
                  .catch((e) => setNostrMsg(e instanceof Error ? e.message : String(e)));
              }}
            >
              Clear key
            </button>
          )}
        </div>
        <label style={{ marginTop: "0.75rem" }}>Import nsec (optional)</label>
        <input
          type="password"
          value={nostrImport}
          onChange={(e) => setNostrImport(e.target.value)}
          placeholder="nsec1…"
          autoComplete="off"
        />
        <div className="form-actions">
          <button
            className="btn"
            disabled={!nostrImport.trim()}
            onClick={() => {
              setNostrMsg(null);
              void api
                .importNostrKey(nostrImport.trim())
                .then((res) => {
                  setNostrImport("");
                  setNostrMsg(`Key imported · ${res.npub}`);
                  return refresh();
                })
                .catch((e) => setNostrMsg(e instanceof Error ? e.message : String(e)));
            }}
          >
            Import key
          </button>
        </div>
        {nostrMsg && <p className="muted-text" style={{ marginTop: "0.5rem" }}>{nostrMsg}</p>}
      </div>
    </div>
  );
}
