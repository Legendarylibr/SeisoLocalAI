import { useEffect, useState } from "react";

type StagePreset = { id: string; label: string; stages: string[] };

type PresetsResponse = {
  presets: StagePreset[];
  stages: string[];
  help: Record<string, string>;
  defaults?: Record<string, string>;
};

export function useStagePipelinePresets(
  fallbackStages: string[],
  loadPresets: () => Promise<PresetsResponse>,
  initialPreset = "smoke",
) {
  const [presets, setPresets] = useState<StagePreset[]>([]);
  const [presetsLoading, setPresetsLoading] = useState(true);
  const [allStages, setAllStages] = useState<string[]>(fallbackStages);
  const [stageHelp, setStageHelp] = useState<Record<string, string>>({});
  const [defaults, setDefaults] = useState<Record<string, string>>({});
  const [preset, setPreset] = useState(initialPreset);
  const [selectedStages, setSelectedStages] = useState<string[]>(fallbackStages);

  const presetList = presets;
  const presetsReady = !presetsLoading && presets.length > 0;

  useEffect(() => {
    loadPresets()
      .then((r) => {
        setPresets(r.presets);
        setAllStages(r.stages.length ? r.stages : fallbackStages);
        setStageHelp(r.help);
        setDefaults(r.defaults ?? {});
        if (r.presets.length > 0) {
          setPreset((current) => (r.presets.some((p) => p.id === current) ? current : r.presets[0].id));
        }
      })
      .catch(console.error)
      .finally(() => setPresetsLoading(false));
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
    presetsLoading,
    presetsReady,
    allStages,
    stageHelp,
    selectedStages,
    toggleStage,
    defaults,
  };
}
