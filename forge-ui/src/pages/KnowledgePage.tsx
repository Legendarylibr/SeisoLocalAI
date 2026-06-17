import { useCallback, useEffect, useState } from "react";
import { api, KnowledgeBase, KnowledgeChunk } from "@/lib/api";
import { StudioPageShell } from "@/components/StudioPageShell";
import { FormSection } from "@/components/research/FormSection";
import { DataTable } from "@/components/research/DataTable";
import { LogStream } from "@/components/research/LogStream";

export function KnowledgePage() {
  const [bases, setBases] = useState<KnowledgeBase[]>([]);
  const [selectedKb, setSelectedKb] = useState("");
  const [newKbId, setNewKbId] = useState("");
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<KnowledgeChunk[]>([]);
  const [logs, setLogs] = useState<string[]>([]);
  const [uploading, setUploading] = useState(false);
  const [ingesting, setIngesting] = useState(false);
  const [searching, setSearching] = useState(false);
  const [lastUpload, setLastUpload] = useState<{ path: string; filename: string } | null>(null);

  const refreshBases = useCallback(() => {
    api.listKnowledgeBases()
      .then((r) => {
        setBases(r.bases);
        setSelectedKb((prev) => prev || (r.bases.length > 0 ? r.bases[0].id : ""));
      })
      .catch(console.error);
  }, []);

  useEffect(() => {
    refreshBases();
  }, [refreshBases]);

  const log = (msg: string) => setLogs((l) => [...l, msg]);

  const createBase = async () => {
    const id = newKbId.trim();
    if (!id) return;
    try {
      await api.createKnowledgeBase(id);
      log(`Created knowledge base: ${id}`);
      setNewKbId("");
      setSelectedKb(id);
      refreshBases();
    } catch (e) {
      log(`Error: ${e instanceof Error ? e.message : String(e)}`);
    }
  };

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    try {
      const res = await api.uploadKnowledgeFile(file);
      setLastUpload({ path: res.path, filename: res.filename });
      log(`Uploaded ${res.filename} (${Math.round(res.size / 1024)} KB)`);
    } catch (err) {
      log(`Upload failed: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setUploading(false);
      e.target.value = "";
    }
  };

  const ingest = async () => {
    if (!selectedKb || !lastUpload) return;
    setIngesting(true);
    try {
      const res = await api.ingestKnowledge(selectedKb, lastUpload.path);
      log(`Ingested ${res.chunk_count ?? 0} chunks into ${selectedKb}`);
      refreshBases();
    } catch (err) {
      log(`Ingest failed: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setIngesting(false);
    }
  };

  const search = async () => {
    if (!selectedKb || !query.trim()) return;
    setSearching(true);
    setResults([]);
    try {
      const res = await api.retrieveKnowledge(selectedKb, query.trim(), 8);
      setResults(res.results);
      log(`Retrieved ${res.results.length} chunks for "${query.trim()}"`);
    } catch (err) {
      log(`Retrieve failed: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setSearching(false);
    }
  };

  return (
    <StudioPageShell
      title="Knowledge Base"
      subtitle="Local RAG corpus — ingest documents, chunk on-device, and retrieve with keyword scoring. No embeddings leave this machine."
      group="Models"
      badge={
        <span className="trust-badge trust-badge-dim">Keyword retrieval · 512-token chunks</span>
      }
    >
      <div className="train-layout">
        <div className="card research-config-card">
          <FormSection title="Corpus" hint="Create and select a knowledge base identifier.">
            <label>Active knowledge base</label>
            <select value={selectedKb} onChange={(e) => setSelectedKb(e.target.value)}>
              <option value="">Select…</option>
              {bases.map((b) => (
                <option key={b.id} value={b.id}>
                  {b.id} ({b.chunk_count} chunks)
                </option>
              ))}
            </select>

            <label>New knowledge base ID</label>
            <div className="field-inline">
              <input
                value={newKbId}
                onChange={(e) => setNewKbId(e.target.value)}
                placeholder="my-research-corpus"
                pattern="[a-zA-Z0-9_-]+"
              />
              <button type="button" className="btn" onClick={createBase} disabled={!newKbId.trim()}>
                Create
              </button>
            </div>
          </FormSection>

          <FormSection title="Ingest" hint="Upload text files, then ingest into the selected corpus." collapsible defaultOpen>
            <label className="file-upload-label">
              <input type="file" accept=".txt,.md,.json,.jsonl,.csv" onChange={handleUpload} disabled={uploading} />
              <span className="file-upload-btn">
                {uploading ? "Uploading…" : "Choose file"}
              </span>
            </label>
            {lastUpload && (
              <p className="field-hint">
                Ready: <code>{lastUpload.filename}</code>
              </p>
            )}
            <button
              type="button"
              className="btn btn-primary btn-lg"
              onClick={ingest}
              disabled={!selectedKb || !lastUpload || ingesting}
            >
              {ingesting ? "Ingesting…" : "Ingest into corpus"}
            </button>
          </FormSection>
        </div>

        <div className="card research-config-card">
          <FormSection title="Retrieve" hint="Keyword overlap scoring — top-k chunks returned.">
            <label>Query</label>
            <textarea
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              rows={3}
              placeholder="What are you looking for in the corpus?"
            />
            <button
              type="button"
              className="btn btn-primary"
              onClick={search}
              disabled={!selectedKb || !query.trim() || searching}
            >
              {searching ? "Searching…" : "Retrieve chunks"}
            </button>
          </FormSection>

          {results.length > 0 && (
            <div className="retrieve-results">
              {results.map((r) => (
                <article key={r.id} className="retrieve-chunk">
                  <header className="retrieve-chunk-head">
                    <span className="retrieve-chunk-source">{r.source}</span>
                    <span className="retrieve-chunk-idx">#{r.chunk_index}</span>
                  </header>
                  <p className="retrieve-chunk-text">{r.text}</p>
                </article>
              ))}
            </div>
          )}

          <LogStream title="Activity" logs={logs} tall />
        </div>
      </div>

      <div className="card">
        <h3 className="section-title">Corpora</h3>
        <DataTable
          columns={[
            { key: "id", header: "ID", mono: true },
            { key: "chunk_count", header: "Chunks" },
            {
              key: "has_index",
              header: "Indexed",
              render: (row) => row.has_index ? "yes" : "—",
            },
          ]}
          rows={bases}
          getRowKey={(b) => b.id}
          emptyMessage="No knowledge bases yet. Create one above."
        />
      </div>
    </StudioPageShell>
  );
}
