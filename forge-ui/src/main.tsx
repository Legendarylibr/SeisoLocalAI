import React from "react";
import ReactDOM from "react-dom/client";
import { App } from "./App";
import { SeisoLogoMark } from "@/components/SeisoLogo";
import "./styles.css";

// Tauri desktop webview: the embedded UI is served from the tauri:// custom
// protocol, but the Forge backend (and its auth cookies) live on
// http://127.0.0.1:8765 — a different origin. SameSite=Strict session cookies
// are never sent cross-origin, so every authenticated request would 401.
// Navigate the webview to the backend origin instead: the backend serves the
// same built app, making auth cookies same-origin. Poll /health first so the
// window does not hit a connection-refused page while the sidecar boots.
const TAURI_BACKEND = "http://127.0.0.1:8765";
const HEALTH_POLL_ATTEMPTS = 120;
const HEALTH_POLL_INTERVAL_MS = 500;

function isTauriEmbeddedOrigin(): boolean {
  if (typeof window === "undefined" || !("__TAURI_INTERNALS__" in window)) {
    return false;
  }
  return (
    window.location.protocol === "tauri:" ||
    window.location.hostname === "tauri.localhost"
  );
}

function renderApp() {
  ReactDOM.createRoot(document.getElementById("root")!).render(
    <React.StrictMode>
      <App />
    </React.StrictMode>,
  );
}

function renderLoading() {
  ReactDOM.createRoot(document.getElementById("root")!).render(
    <div className="app-loading">
      <div className="app-loading-atmosphere" aria-hidden />
      <div className="app-loading-mark app-loading-mark-wordmark">
        <SeisoLogoMark className="app-loading-logo" />
      </div>
      <div className="app-loading-bar" aria-hidden />
      <p className="app-loading-text">Starting Seiso…</p>
    </div>,
  );
}

function renderBackendUnreachable() {
  ReactDOM.createRoot(document.getElementById("root")!).render(
    <div className="app-loading">
      <div className="app-loading-atmosphere" aria-hidden />
      <div className="app-loading-mark app-loading-mark-wordmark">
        <SeisoLogoMark className="app-loading-logo" />
      </div>
      <p className="app-loading-text">
        Could not reach the Seiso backend at {TAURI_BACKEND}. Close and start again, or run
        `seiso forge` in a terminal to diagnose.
      </p>
    </div>,
  );
}

if (isTauriEmbeddedOrigin()) {
  renderLoading();
  (async () => {
    for (let attempt = 0; attempt < HEALTH_POLL_ATTEMPTS; attempt++) {
      try {
        const res = await fetch(`${TAURI_BACKEND}/health`);
        if (res.ok) {
          window.location.replace(TAURI_BACKEND);
          return;
        }
      } catch {
        // backend not up yet — retry
      }
      // Promise.withResolvers requires lib ES2024; project targets ES2022.
      await new Promise<void>((resolve) => setTimeout(resolve, HEALTH_POLL_INTERVAL_MS));
    }
    renderBackendUnreachable();
  })();
} else {
  renderApp();
}
