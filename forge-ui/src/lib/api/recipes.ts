import { request } from "./client";

export const recipesApi = {
  runRecipe: (recipe: Record<string, unknown>) =>
    request<{ job_id: string }>("/recipes/jobs", {
      method: "POST",
      body: JSON.stringify({ recipe }),
    }),
};
