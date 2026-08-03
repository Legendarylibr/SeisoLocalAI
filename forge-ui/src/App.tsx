import { lazy, Suspense, useState } from "react";
import { BrowserRouter, Navigate, Route, Routes } from "react-router";
import { AuthProvider, useAuth } from "@/hooks/useAuth";
import { HardwareProfileProvider } from "@/context/HardwareProfileContext";
import { MetricsProvider } from "@/context/MetricsContext";
import { PlatformSettingsProvider, usePlatformSettings } from "@/context/PlatformSettingsContext";
import { TrainingModelsProvider } from "@/context/TrainingModelsContext";
import { Layout } from "@/components/Layout";
import { SeisoLogoMark } from "@/components/SeisoLogo";
import { AuthPage } from "@/pages/AuthPage";
import { HfTokenPage } from "@/pages/HfTokenPage";
import { needsHfTokenOnboarding, skipHfOnboarding } from "@/lib/hfOnboarding";

const DashboardPage = lazy(() => import("@/pages/DashboardPage").then((m) => ({ default: m.DashboardPage })));
const HubPage = lazy(() => import("@/pages/HubPage").then((m) => ({ default: m.HubPage })));
const ChatPage = lazy(() => import("@/pages/ChatPage").then((m) => ({ default: m.ChatPage })));
const ExportPage = lazy(() => import("@/pages/ExportPage").then((m) => ({ default: m.ExportPage })));
const CompressPage = lazy(() => import("@/pages/CompressPage").then((m) => ({ default: m.CompressPage })));
const DistillRLPage = lazy(() => import("@/pages/DistillRLPage").then((m) => ({ default: m.DistillRLPage })));
const TrainPage = lazy(() => import("@/pages/TrainPage").then((m) => ({ default: m.TrainPage })));
const KnowledgePage = lazy(() => import("@/pages/KnowledgePage").then((m) => ({ default: m.KnowledgePage })));
const RecipesPage = lazy(() => import("@/pages/RecipesPage").then((m) => ({ default: m.RecipesPage })));
const IntegrationsPage = lazy(() =>
  import("@/pages/IntegrationsPage").then((m) => ({ default: m.IntegrationsPage })),
);
const SettingsPage = lazy(() => import("@/pages/SettingsPage").then((m) => ({ default: m.SettingsPage })));

function PageLoading() {
  return (
    <div className="app-loading">
      <div className="app-loading-atmosphere" aria-hidden />
      <div className="app-loading-mark app-loading-mark-wordmark">
        <SeisoLogoMark className="app-loading-logo" />
      </div>
      <div className="app-loading-bar" aria-hidden />
      <p className="app-loading-text">Seiso Local AI</p>
    </div>
  );
}

function Guard({
  children,
  fullBleed = false,
  dismissedHfOnboardingUserId,
  onDismissHfOnboarding,
}: {
  children: React.ReactNode;
  fullBleed?: boolean;
  dismissedHfOnboardingUserId: string | null;
  onDismissHfOnboarding: (userId: string) => void;
}) {
  const { user, loading, needsOnboarding, keyBackup } = useAuth();
  const { hfStatus, loading: platformLoading } = usePlatformSettings();

  const dismissHfOnboarding = () => {
    if (!user) return;
    skipHfOnboarding(user.id);
    onDismissHfOnboarding(user.id);
  };

  if (loading) {
    return <PageLoading />;
  }
  // Fresh keygen: stay on AuthPage until the user confirms they wrote the nsec down.
  if (keyBackup) return <AuthPage />;
  if (!user && !needsOnboarding) return <AuthPage />;
  if (needsOnboarding && !user) return <AuthPage />;
  if (platformLoading && user) return <PageLoading />;
  if (
    user &&
    dismissedHfOnboardingUserId !== user.id &&
    needsHfTokenOnboarding(hfStatus, user.id)
  ) {
    return <HfTokenPage onDone={dismissHfOnboarding} />;
  }
  return (
    <Layout fullBleed={fullBleed}>
      <Suspense fallback={<PageLoading />}>{children}</Suspense>
    </Layout>
  );
}

function AppRoutes() {
  const [dismissedHfOnboardingUserId, setDismissedHfOnboardingUserId] = useState<string | null>(null);
  const guardProps = {
    dismissedHfOnboardingUserId,
    onDismissHfOnboarding: setDismissedHfOnboardingUserId,
  };

  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Guard {...guardProps}><DashboardPage /></Guard>} />
        <Route path="/hub" element={<Guard {...guardProps}><HubPage /></Guard>} />
        <Route path="/chat" element={<Guard {...guardProps} fullBleed><ChatPage /></Guard>} />
        <Route path="/export" element={<Guard {...guardProps}><ExportPage /></Guard>} />
        <Route path="/compress" element={<Guard {...guardProps}><CompressPage /></Guard>} />
        <Route path="/distill-rl" element={<Guard {...guardProps}><DistillRLPage /></Guard>} />
        <Route path="/train" element={<Guard {...guardProps}><TrainPage /></Guard>} />
        <Route path="/knowledge" element={<Guard {...guardProps}><KnowledgePage /></Guard>} />
        <Route path="/recipes" element={<Guard {...guardProps}><RecipesPage /></Guard>} />
        <Route path="/integrations" element={<Guard {...guardProps}><IntegrationsPage /></Guard>} />
        <Route path="/settings" element={<Guard {...guardProps}><SettingsPage /></Guard>} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}

export function App() {
  return (
    <AuthProvider>
      <PlatformSettingsProvider>
        <HardwareProfileProvider>
          <TrainingModelsProvider>
            <MetricsProvider>
              <AppRoutes />
            </MetricsProvider>
          </TrainingModelsProvider>
        </HardwareProfileProvider>
      </PlatformSettingsProvider>
    </AuthProvider>
  );
}
