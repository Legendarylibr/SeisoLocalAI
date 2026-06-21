/** Build Ollama import commands from completed export job output paths. */
export function buildOllamaCommandsFromExport(
  outputPathsJson: string | undefined,
  modelName: string,
): string[] {
  if (!outputPathsJson) return [];
  let outputs: Record<string, string>;
  try {
    outputs = JSON.parse(outputPathsJson) as Record<string, string>;
  } catch {
    return [];
  }

  const safeName = modelName.trim().replace(/\s+/g, "-") || "seiso-model";
  const ggufEntries = Object.entries(outputs).filter(([key]) => key.toLowerCase().includes("gguf"));
  const commands: string[] = [];

  for (const [key, rawPath] of ggufEntries) {
    const dir = rawPath.replace(/[/\\][^/\\]+$/, "");
    const quant = key.startsWith("gguf_") ? key.slice("gguf_".length) : key;
    const name = ggufEntries.length === 1 ? safeName : `${safeName}-${quant}`;
    commands.push(`ollama create ${name} -f ${dir}/Modelfile`);
  }

  return commands;
}

export function extractOllamaCommandsFromLogs(logs: string[]): string[] {
  return logs
    .filter((line) => line.startsWith("Ollama: "))
    .map((line) => line.slice("Ollama: ".length).trim())
    .filter(Boolean);
}
