/** Client-side mirror of seiso.models.trainable_snapshot.is_gguf_only_repo_id. */

const GGUF_REPO_RE = /(?:^|\/)[^/]*-gguf(?:$|\/|-)/i;

const TRUSTED_GGUF_PUBLISHERS = new Set([
  "unsloth",
  "bartowski",
  "QuantFactory",
  "lmstudio-community",
  "TheBloke",
  "mradermacher",
  "MaziyarPanahi",
  "mlx-community",
]);

export const GGUF_TRAIN_ERROR =
  "This repo is GGUF-only (chat/inference weights). LoRA/QLoRA training needs a safetensors or PyTorch checkpoint.";

function repoOwner(repoId: string): string {
  const slash = repoId.indexOf("/");
  return slash >= 0 ? repoId.slice(0, slash) : repoId;
}

function isTrustedGgufRepo(repoId: string): boolean {
  const owner = repoOwner(repoId);
  if (TRUSTED_GGUF_PUBLISHERS.has(owner)) {
    const lower = repoId.toLowerCase();
    return lower.includes("gguf") || GGUF_REPO_RE.test(lower) || lower.endsWith("-gguf");
  }
  return false;
}

export function isGgufOnlyRepoId(
  repoId: string,
  tags: string[] | readonly string[] = [],
): boolean {
  const lowered = repoId.trim().toLowerCase();
  if (!lowered) return false;
  if (GGUF_REPO_RE.test(lowered) || lowered.endsWith("-gguf")) return true;
  if (isTrustedGgufRepo(repoId)) return true;
  const tagSet = new Set(tags.map((t) => t.toLowerCase()));
  return tagSet.has("gguf") && !tagSet.has("safetensors") && !tagSet.has("pytorch");
}