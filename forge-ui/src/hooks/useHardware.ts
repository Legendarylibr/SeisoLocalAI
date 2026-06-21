import { useHardwareProfileContext } from "@/context/HardwareProfileContext";

/** Cached local hardware profile — fetched once per authenticated session. */
export function useHardwareProfile() {
  const { profile, loading, refresh } = useHardwareProfileContext();
  return { profile, loading, refresh };
}
