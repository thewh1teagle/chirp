import { invoke } from "@tauri-apps/api/core";
import { motion } from "framer-motion";
import { Loader2 } from "lucide-react";
import { useEffect, useState } from "react";
import { Navigate, Route, Routes, useLocation, useNavigate } from "react-router-dom";
import "./App.css";
import { AgentsPage, HomePage, OnboardPage, SettingsPage } from "./components/pages";
import { Brand } from "./components/ui";
import { ModelBundle, StudioState } from "./types";
import { sampleText } from "./utils";

function App() {
  const location = useLocation();
  const navigate = useNavigate();
  const [bundle, setBundle] = useState<ModelBundle | null>(null);
  const [checking, setChecking] = useState(true);
  const [studio, setStudio] = useState<StudioState>({
    text: sampleText,
    referencePath: "",
    languages: ["auto"],
    language: "auto",
    audioPath: "",
    step: "idle",
    status: "Ready to generate.",
    busy: false,
    error: "",
  });

  useEffect(() => {
    let cancelled = false;

    async function boot() {
      try {
        const current = await invoke<ModelBundle>("get_model_bundle");
        if (cancelled) return;

        setBundle(current);
        if (current.installed && ["/", "/onboard"].includes(location.pathname)) navigate("/home", { replace: true });
        if (!current.installed && location.pathname !== "/onboard") navigate("/onboard", { replace: true });
      } catch {
        if (!cancelled && location.pathname !== "/onboard") navigate("/onboard", { replace: true });
      } finally {
        if (!cancelled) setChecking(false);
      }
    }

    boot();
    return () => {
      cancelled = true;
    };
  }, [location.pathname, navigate]);

  if (checking) {
    return (
      <main className="grid min-h-screen place-items-center bg-background p-8 text-primary">
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex flex-col items-center space-y-6 text-center"
        >
          <Brand className="scale-150" />
          <div className="flex items-center gap-3 text-secondary">
            <Loader2 className="h-4 w-4 animate-spin" />
            <p className="text-sm font-medium tracking-wide">Preparing your local voice studio...</p>
          </div>
        </motion.div>
      </main>
    );
  }

  return (
    <Routes>
      <Route path="/onboard" element={<OnboardPage bundle={bundle} setBundle={setBundle} />} />
      <Route path="/home" element={<HomePage bundle={bundle} setBundle={setBundle} studio={studio} setStudio={setStudio} />} />
      <Route path="/agents" element={<AgentsPage bundle={bundle} />} />
      <Route path="/settings" element={<SettingsPage bundle={bundle} />} />
      <Route path="*" element={<Navigate to={bundle?.installed ? "/home" : "/onboard"} replace />} />
    </Routes>
  );
}

export default App;
