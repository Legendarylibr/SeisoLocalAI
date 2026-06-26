import type { HfAuthInfo, HfHubStatus } from "@/lib/api";

type PublishAuthOptions = {
  requestToken?: string;
  useCli?: boolean;
};

/** True when Hub upload/publish can proceed (valid saved/env/CLI token or per-request token). */
export function canPublishToHub(
  hfStatus: HfHubStatus | null,
  hfAuth: HfAuthInfo | null | undefined,
  options: PublishAuthOptions = {},
): boolean {
  if (options.requestToken?.trim()) return true;
  if (options.useCli && hfAuth?.cli_logged_in) return true;
  return hfStatus?.ready_for_upload === true || hfStatus?.connectivity.token_valid === true;
}