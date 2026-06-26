import type { HfHubStatus } from "@/lib/api";

export const hfOnboardingSkipKey = (userId: string) => `seiso_hf_prompt_skipped:${userId}`;

export function isHfOnboardingSkipped(userId: string): boolean {
  return sessionStorage.getItem(hfOnboardingSkipKey(userId)) === "1";
}

export function skipHfOnboarding(userId: string): void {
  sessionStorage.setItem(hfOnboardingSkipKey(userId), "1");
}

/** True when the post-login HF token screen should appear (skippable, non-blocking). */
export function needsHfTokenOnboarding(
  hfStatus: HfHubStatus | null,
  userId: string,
): boolean {
  if (isHfOnboardingSkipped(userId)) return false;
  if (hfStatus?.connectivity.token_valid) return false;
  return true;
}