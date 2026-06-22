/** Heuristic: GGUF inference mirrors cannot be used as LoRA/SFT base models. */
export function isGgufOnlyRepoId(repoId: string): boolean {
  const id = repoId.trim();
  if (!id) return false;
  if (/-gguf$/i.test(id)) return true;
  if (/(?:^|\/)unsloth\/[^/]+-gguf/i.test(id)) return true;
  if (/(?:^|\/)bartowski\/[^/]+-gguf/i.test(id)) return true;
  if (/(?:^|\/)QuantFactory\/[^/]+-gguf/i.test(id)) return true;
  if (/(?:^|\/)lmstudio-community\/[^/]+-gguf/i.test(id)) return true;
  return false;
}

export const GGUF_TRAIN_ERROR =
  "GGUF repos are for chat/inference only. Pick a safetensors checkpoint (e.g. Qwen/Qwen2.5-0.5B-Instruct).";
