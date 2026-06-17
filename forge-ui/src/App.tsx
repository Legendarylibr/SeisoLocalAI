import { lazy, Suspense, useEffect, useState } from "react";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { AuthProvider, useAuth } from "@/hooks/useAuth";
import { Layout } from "@/components/Layout";
import { SeisoLogoMark } from "@/components/SeisoLogo";
import { HfTokenPrompt } from "@/components/HfTokenPrompt";
import { AuthPage } from "@/pages/AuthPage";
import { api } from "@/lib/api";

const DashboardPage = lazy(() => import("@/pages/DashboardPage").then((m) => ({ default: m.DashboardPage })));
const HubPage = lazy(() => import("@/pages/HubPage").then((m) => ({ default: m.HubPage })));
const ChatPage = lazy(() => import("@/pages/ChatPage").then((m) => ({ default: m.ChatPage })));
const TrainPage = lazy(() => import("@/pages/TrainPage").then((m) => ({ default: m.TrainPage })));
const ExportPage = lazy(() => import("@/pages/ExportPage").then((m) => ({ default: m.ExportPage })));
const RLQuantPage = lazy(() => import("@/pages/RLQuantPage").then((m) => ({ default: m.RLQuantPage })));
const CompressPage = lazy(() => import("@/pages/CompressPage").then((m) => ({ default: m.CompressPage })));
const ImageCompressPage = lazy(() =>
  import("@/pages/ImageCompressPage").then((m) => ({ default: m.ImageCompressPage })),
);
const KnowledgePage = lazy(() => import("@/pages/KnowledgePage").then((m) => ({ default: m.KnowledgePage })));
const RecipesPage = lazy(() => import("@/pages/RecipesPage").then((m) => ({ default: m.RecipesPage })));
const IntegrationsPage = lazy(() =>
  import("@/pages/IntegrationsPage").then((m) => ({ default: m.IntegrationsPage })),
);
const SettingsPage = lazy(() => import("@/pages/SettingsPage").then((m) => ({ default: m.SettingsPage })));

function PageLoading() {
  return (
    <div className="app-loading">
      <div className="app-loading-mark">
        <SeisoLogoMark className="brand-logo-img app-loading-logo" />
      </div>
      <div className="app-loading-bar" aria-hidden />
      <p className="app-loading-text">Seiso Local AI</p>
    </div>
  );
}

function Guard({ children, fullBleed = false }: { children: React.ReactNode; fullBleed?: boolean }) {
  const { user, loading, needsOnboarding } = useAuth();
  const [checkingHfPrompt, setCheckingHfPrompt] = useState(false);
  const [showHfPrompt, setShowHfPrompt] = useState(false);

  useEffect(() => {
    if (!user) {
      setShowHfPrompt(false);
      setCheckingHfPrompt(false);
      return;
    }
    const key = `seiso_hf_prompt_skipped:${user.id}`;
    if (sessionStorage.getItem(key) === "1") {
      setShowHfPrompt(false);
      setCheckingHfPrompt(false);
      return;
    }
    setCheckingHfPrompt(true);
    api.settings()
      .then((settings) => {
        setShowHfPrompt(!settings.hf_configured && !settings.hf_auth.user_token_saved);
      })
      .catch(() => setShowHfPrompt(true))
      .finally(() => setCheckingHfPrompt(false));
  }, [user]);

  const dismissHfPrompt = () => {
    if (user) sessionStorage.setItem(`seiso_hf_prompt_skipped:${user.id}`, "1");
    setShowHfPrompt(false);
  };

  if (loading) {
    return (
      <div className="app-loading">
        <div className="app-loading-mark">
          <SeisoLogoMark className="brand-logo-img app-loading-logo" />
        </div>
        <div className="app-loading-bar" aria-hidden />
        <p className="app-loading-text">Seiso Local AI</p>
      </div>
    );
  }
  if (!user && !needsOnboarding) return <AuthPage />;
  if (needsOnboarding && !user) return <AuthPage />;
  if (checkingHfPrompt) return <PageLoading />;
  if (showHfPrompt) return <HfTokenPrompt onDone={dismissHfPrompt} />;
  return (
    <Layout fullBleed={fullBleed}>
      <Suspense fallback={<PageLoading />}>{children}</Suspense>
    </Layout>
  );
}

export function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Guard><DashboardPage /></Guard>} />
          <Route path="/hub" element={<Guard><HubPage /></Guard>} />
          <Route path="/chat" element={<Guard fullBleed><ChatPage /></Guard>} />
          <Route path="/train" element={<Guard><TrainPage /></Guard>} />
          <Route path="/export" element={<Guard><ExportPage /></Guard>} />
          <Route path="/rl-quant" element={<Guard><RLQuantPage /></Guard>} />
          <Route path="/compress" element={<Guard><CompressPage /></Guard>} />
          <Route path="/image-compress" element={<Guard><ImageCompressPage /></Guard>} />
          <Route path="/knowledge" element={<Guard><KnowledgePage /></Guard>} />
          <Route path="/recipes" element={<Guard><RecipesPage /></Guard>} />
          <Route path="/integrations" element={<Guard><IntegrationsPage /></Guard>} />
          <Route path="/settings" element={<Guard><SettingsPage /></Guard>} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  );
}
