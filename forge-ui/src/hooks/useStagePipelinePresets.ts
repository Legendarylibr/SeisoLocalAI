import { useEffect, useState } from "react";

type StagePreset = { id: string; label: string; stages: string[] };

export function useStagePipelinePresets(
  fallbackPresets: StagePreset[],
  fallbackStages: string[],
  loadPresets: () => Promise<{ presets: StagePreset[]; stages: string[]; help: Record<string, string> }>,
  initialPreset = fallbackPresets[0]?.id ?? "smoke",
) {
  const [presets, setPresets] = useState<StagePreset[]>([]);
  const [allStages, setAllStages] = useState<string[]>(fallbackStages);
  const [stageHelp, setStageHelp] = useState<Record<string, string>>({});
  const [preset, setPreset] = useState(initialPreset);
  const [selectedStages, setSelectedStages] = useState<string[]>(
    fallbackPresets[0]?.stages.length ? fallbackPresets[0].stages : fallbackStages,
  );

  const presetList = presets.length ? presets : fallbackPresets;

  useEffect(() => {
    loadPresets()
      .then((r) => {
        setPresets(r.presets);
        setAllStages(r.stages.length ? r.stages : fallbackStages);
        setStageHelp(r.help);
      })
      .catch(console.error);
    // Load presets once on mount; loader is stable for each page.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const p = presetList.find((x) => x.id === preset);
    if (!p) return;
    setSelectedStages(p.stages.length ? p.stages : allStages);
  }, [preset, presetList, allStages]);

  const toggleStage = (stage: string) => {
    setSelectedStages((prev) =>
      prev.includes(stage) ? prev.filter((s) => s !== stage) : [...prev, stage],
    );
  };

  return {
    preset,
    setPreset,
    presetList,
    allStages,
    stageHelp,
    selectedStages,
    toggleStage,
  };
}
