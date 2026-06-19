import { useEffect, useRef, useState } from "react";
import { RecipeCanvas, RecipeGraph } from "@/components/RecipeCanvas";
import { LogStream } from "@/components/research/LogStream";
import { StudioPageShell } from "@/components/StudioPageShell";
import { api, subscribeSSE } from "@/lib/api";

export function RecipesPage() {
  const [logs, setLogs] = useState<string[]>([]);
  const [recipe, setRecipe] = useState<RecipeGraph | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
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
      setError("Select the Import node and set a file path under your uploads folder.");
      return;
    }

    setError(null);
    setLogs([]);
    setRunning(true);
    sseAbortRef.current?.();
    try {
      const res = await api.runRecipe(r);
      sseAbortRef.current = subscribeSSE(
        `/recipes/jobs/${res.job_id}/stream`,
        (event, data) => {
          if (event === "error") setLogs((l) => [...l, `ERROR: ${data}`]);
          else if (event === "done") setRunning(false);
          else setLogs((l) => [...l, data]);
        },
        (err) => {
          setError(err.message);
          setRunning(false);
        },
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to start recipe");
      setRunning(false);
    }
  };

  return (
    <StudioPageShell
      title="Recipe Studio"
      subtitle="Build a visual data pipeline — import, transform, filter, sample, and export."
      className="recipe-studio-page"
    >
      <div className="card recipe-studio-card">
        <RecipeCanvas
          onChange={(r) => {
            recipeRef.current = r;
            setRecipe(r);
          }}
        />
      </div>

      <div className="recipe-run-bar card">
        <div className="recipe-run-meta">
          {recipe ? (
            <>
              <strong>{recipe.name}</strong>
              <span className="muted-text">
                {recipe.nodes.length} nodes · {recipe.edges.length} connections
              </span>
            </>
          ) : (
            <span className="muted-text">Connect nodes and configure the Import path to run.</span>
          )}
        </div>
        <button className="btn btn-primary" onClick={run} disabled={running}>
          {running ? "Running…" : "Run recipe"}
        </button>
        {error && <p className="error-text">{error}</p>}
      </div>

      {(logs.length > 0 || running) && (
        <div className="card">
          <LogStream title="Pipeline output" logs={logs} tall />
        </div>
      )}
    </StudioPageShell>
  );
}
