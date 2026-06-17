import { useRef, useState } from "react";
import { RecipeCanvas, RecipeGraph } from "@/components/RecipeCanvas";
import { api, subscribeSSE } from "@/lib/api";

export function RecipesPage() {
  const [logs, setLogs] = useState<string[]>([]);
  const [recipe, setRecipe] = useState<RecipeGraph | null>(null);
  const recipeRef = useRef<RecipeGraph | null>(null);

  const run = async () => {
    const r = recipeRef.current || recipe;
    if (!r) {
      alert("Build your recipe on the canvas first (click Serialize).");
      return;
    }
    setLogs([]);
    const res = await api.runRecipe(r);
    subscribeSSE(`/recipes/jobs/${res.job_id}/stream`, (_, data) => {
      setLogs((l) => [...l, data]);
    });
  };

  return (
    <div>
      <h1 className="page-title">Recipe Studio</h1>
      <p className="page-sub">Drag-and-drop node workflow to build and transform datasets.</p>

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
      </div>

      {logs.length > 0 && (
        <div className="card">
          <div className="log-panel">{logs.join("\n")}</div>
        </div>
      )}
    </div>
  );
}
