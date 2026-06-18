import { useHardwareProfileContext } from "@/context/HardwareProfileContext";

/** Cached local hardware profile — fetched once per authenticated session. */
export function useHardwareProfile() {
  const { profile, loading } = useHardwareProfileContext();
  return { profile, loading };
}
