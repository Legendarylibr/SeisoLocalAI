import { useEffect, useState } from "react";
import { api, HardwareProfile } from "@/lib/api";

/** One-shot local hardware profile — never leaves the machine. */
export function useHardwareProfile() {
  const [profile, setProfile] = useState<HardwareProfile | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api
      .hardware()
      .then(setProfile)
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  return { profile, loading };
}
