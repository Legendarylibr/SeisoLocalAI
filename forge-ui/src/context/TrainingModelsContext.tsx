import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { api, TrainableModel } from "@/lib/api";
import { invalidateApiCache, refreshCachedGet } from "@/lib/api/getCache";
import { useAuth } from "@/hooks/useAuth";

type TrainingModelsState = {
  models: TrainableModel[];
  loading: boolean;
  refresh: () => Promise<TrainableModel[]>;
};

const TrainingModelsContext = createContext<TrainingModelsState | null>(null);

export function TrainingModelsProvider({ children }: { children: ReactNode }) {
  const { user, loading: authLoading } = useAuth();
  const [models, setModels] = useState<TrainableModel[]>([]);
  const [loading, setLoading] = useState(false);
  const cacheRef = useRef<{ userId: string; models: TrainableModel[] } | null>(null);

  const refresh = useCallback(async () => {
    if (!user) {
      setModels([]);
      return [];
    }
    setLoading(true);
    try {
      invalidateApiCache("/training/models");
      const resp = await refreshCachedGet<{ models: TrainableModel[]; total: number }>(
        "/training/models",
        120_000,
      );
      cacheRef.current = { userId: user.id, models: resp.models };
      setModels(resp.models);
      return resp.models;
    } catch {
      return cacheRef.current?.models ?? [];
    } finally {
      setLoading(false);
    }
  }, [user]);

  useEffect(() => {
    if (authLoading) return;
    if (!user) {
      setModels([]);
      cacheRef.current = null;
      setLoading(false);
      return;
    }
    const cached = cacheRef.current;
    if (cached?.userId === user.id) {
      setModels(cached.models);
      return;
    }
    setLoading(true);
    api
      .listTrainingModels()
      .then((resp) => {
        cacheRef.current = { userId: user.id, models: resp.models };
        setModels(resp.models);
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [user, authLoading]);

  return (
    <TrainingModelsContext.Provider value={{ models, loading, refresh }}>
      {children}
    </TrainingModelsContext.Provider>
  );
}

export function useTrainingModels(): TrainingModelsState {
  const ctx = useContext(TrainingModelsContext);
  if (!ctx) {
    throw new Error("useTrainingModels must be used within TrainingModelsProvider");
  }
  return ctx;
}
