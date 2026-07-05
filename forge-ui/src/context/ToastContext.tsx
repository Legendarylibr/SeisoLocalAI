import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { IconClose, IconShield, IconActivity } from "@/components/Icons";

export type ToastTone = "info" | "success" | "error";

export type Toast = {
  id: number;
  message: string;
  tone: ToastTone;
  detail?: string;
};

type NotifyOptions = { tone?: ToastTone; detail?: string; duration?: number };

type ToastState = {
  notify: (message: string, options?: NotifyOptions) => number;
  dismiss: (id: number) => void;
};

const ToastContext = createContext<ToastState | null>(null);

const DEFAULT_DURATION = 4200;

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);
  const timers = useRef(new Map<number, ReturnType<typeof setTimeout>>());
  const nextId = useRef(1);

  const dismiss = useCallback((id: number) => {
    setToasts((current) => current.filter((t) => t.id !== id));
    const timer = timers.current.get(id);
    if (timer) {
      clearTimeout(timer);
      timers.current.delete(id);
    }
  }, []);

  const notify = useCallback(
    (message: string, options?: NotifyOptions) => {
      const id = nextId.current++;
      const tone = options?.tone ?? "info";
      const duration = options?.duration ?? DEFAULT_DURATION;
      setToasts((current) => [
        ...current.slice(-3),
        { id, message, tone, detail: options?.detail },
      ]);
      if (duration > 0) {
        const timer = setTimeout(() => dismiss(id), duration);
        timers.current.set(id, timer);
      }
      return id;
    },
    [dismiss],
  );

  const value = useMemo(() => ({ notify, dismiss }), [notify, dismiss]);

  return (
    <ToastContext.Provider value={value}>
      {children}
      <div className="toast-stack" role="region" aria-label="Notifications" aria-live="polite">
        {toasts.map((toast) => (
          <div key={toast.id} className={`toast toast-${toast.tone}`} role="status">
            <span className="toast-icon" aria-hidden>
              {toast.tone === "error" ? (
                <IconShield size={16} />
              ) : (
                <IconActivity size={16} />
              )}
            </span>
            <div className="toast-body">
              <span className="toast-message">{toast.message}</span>
              {toast.detail && <span className="toast-detail">{toast.detail}</span>}
            </div>
            <button
              type="button"
              className="toast-close"
              onClick={() => dismiss(toast.id)}
              aria-label="Dismiss notification"
            >
              <IconClose size={14} />
            </button>
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  );
}

export function useToast(): ToastState {
  const ctx = useContext(ToastContext);
  if (!ctx) {
    throw new Error("useToast must be used within a ToastProvider");
  }
  return ctx;
}
