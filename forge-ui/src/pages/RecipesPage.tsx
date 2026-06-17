import { useEffect, useRef, useState } from "react";
import { RecipeCanvas, RecipeGraph } from "@/components/RecipeCanvas";
import { StudioPageShell } from "@/components/StudioPageShell";
import { api, subscribeSSE } from "@/lib/api";

export function RecipesPage() {
  const [logs, setLogs] = useState<string[]>([]);
  const [recipe, setRecipe] = useState<RecipeGraph | null>(null);
  const [error, setError] = useState<string | null>(null);
  const recipeRef = useRef<RecipeGraph | null>(null);
  const sseAbortRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    return () => {
      sseAbortRef.current?.();
    };
  }, []);

  const run = async () => {
    const r = recipeRef.current || recipe;
    if (!r) {
      setError("Build your recipe on the canvas first.");
      return;
    }
    const importNode = r.nodes.find((n) => n.type === "import");
    const importPath = String(importNode?.config?.path ?? "").trim();
    if (!importPath) {
      setError("Set an absolute import file path under your uploads folder before running.");
      return;
    }

    setError(null);
    setLogs([]);
    sseAbortRef.current?.();
    try {
      const res = await api.runRecipe(r);
      sseAbortRef.current = subscribeSSE(
        `/recipes/jobs/${res.job_id}/stream`,
        (event, data) => {
          if (event === "error") setLogs((l) => [...l, `ERROR: ${data}`]);
          else setLogs((l) => [...l, data]);
        },
        (err) => setError(err.message),
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to start recipe");
    }
  };

  return (
    <StudioPageShell
      title="Recipe Studio"
      subtitle="Drag-and-drop node workflow to build and transform datasets."
    >
      <div className="card" style={{ padding: 0, overflow: "hidden" }}>
        <RecipeCanvas
          onChange={(r) => {
            recipeRef.current = r;
            setRecipe(r);
          }}
        />
      </div>

      <div className="card" style={{ marginTop: "1rem" }}>
        <button className="btn btn-primary" onClick={run}>
          Run recipe
        </button>
        {recipe && (
          <p style={{ marginTop: "0.75rem", color: "var(--muted)", fontSize: "0.85rem" }}>
            Ready: {recipe.name} · {recipe.nodes.length} nodes · {recipe.edges.length} edges
          </p>
        )}
        {error && <p className="error-text" style={{ marginTop: "0.75rem" }}>{error}</p>}
      </div>

      {logs.length > 0 && (
        <div className="card">
          <div className="log-panel">{logs.join("\n")}</div>
        </div>
      )}
    </StudioPageShell>
  );
}
