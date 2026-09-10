import React from "react";
import ReactDOM from "react-dom/client";
import { App } from "./App";
import "./styles.css";

// Tauri desktop webview: the embedded UI is served from the tauri:// custom
// protocol, but the Forge backend (and its auth cookies) live on
// http://127.0.0.1:8765 — a different origin. SameSite=Strict session cookies
// are never sent cross-origin, so every authenticated request would 401.
// Navigate the webview to the backend origin instead: the backend serves the
// same built app, making auth cookies same-origin. Poll /health first so the
// window does not hit a connection-refused page while the sidecar boots.
const TAURI_BACKEND = "http://127.0.0.1:8765";

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

if (isTauriEmbeddedOrigin()) {
  // Show the same loading screen the app uses while we wait for the backend.
  ReactDOM.createRoot(document.getElementById("root")!).render(
    <div className="app-loading">
      <div className="app-loading-atmosphere" aria-hidden />
      <div className="app-loading-mark app-loading-mark-wordmark">
        <span className="app-loading-text">Seiso Local AI</span>
      </div>
      <div className="app-loading-bar" aria-hidden />
      <p className="app-loading-text">Starting Seiso…</p>
    </div>,
  );
  (async () => {
    for (let attempt = 0; attempt < 120; attempt++) {
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
      await new Promise<void>((resolve) => setTimeout(resolve, 500));
    }
  })();
} else {
  renderApp();
}
