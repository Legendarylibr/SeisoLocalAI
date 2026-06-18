import { request } from "./client";
import { cachedGet } from "./getCache";
import type { GuideStep, HardwareProfile, SystemMetrics } from "./types";

export const systemApi = {
  hardware: () => cachedGet<HardwareProfile>("/system/hardware", 120_000),
  metrics: () => request<SystemMetrics>("/system/metrics"),
  guide: (goal: string) =>
    request<{ goal: string; steps: GuideStep[]; hardware_summary: Record<string, unknown>; local_only: boolean }>(
      `/system/guide?goal=${goal}`,
    ),
};
