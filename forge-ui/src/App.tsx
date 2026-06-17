import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { AuthProvider, useAuth } from "@/hooks/useAuth";
import { Layout } from "@/components/Layout";
import { AuthPage } from "@/pages/AuthPage";
import { HubPage } from "@/pages/HubPage";
import { ChatPage } from "@/pages/ChatPage";
import { TrainPage } from "@/pages/TrainPage";
import { ExportPage } from "@/pages/ExportPage";
import { RecipesPage } from "@/pages/RecipesPage";
import { IntegrationsPage } from "@/pages/IntegrationsPage";
import { SettingsPage } from "@/pages/SettingsPage";

function Guard({ children }: { children: React.ReactNode }) {
  const { user, loading, needsOnboarding } = useAuth();
  if (loading) return <div className="auth-page">Loading…</div>;
  if (!user && !needsOnboarding) return <AuthPage />;
  if (needsOnboarding && !user) return <AuthPage />;
  return <Layout>{children}</Layout>;
}

export function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Guard><HubPage /></Guard>} />
          <Route path="/chat" element={<Guard><ChatPage /></Guard>} />
          <Route path="/train" element={<Guard><TrainPage /></Guard>} />
          <Route path="/export" element={<Guard><ExportPage /></Guard>} />
          <Route path="/recipes" element={<Guard><RecipesPage /></Guard>} />
          <Route path="/integrations" element={<Guard><IntegrationsPage /></Guard>} />
          <Route path="/settings" element={<Guard><SettingsPage /></Guard>} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  );
}
