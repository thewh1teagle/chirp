import { invoke } from "@tauri-apps/api/core";
import { motion } from "framer-motion";
import { Loader2 } from "lucide-react";
import { useEffect, useState } from "react";
import { Navigate, Route, Routes, useLocation, useNavigate } from "react-router-dom";
import "./App.css";
import { Brand } from "./components/ui";
import { AgentsPage } from "./pages/AgentsPage";
import { HomePage } from "./pages/home/HomePage";
import { OnboardPage } from "./pages/OnboardPage";
import { SettingsPage } from "./pages/SettingsPage";
import { ModelBundle, StudioState } from "./lib/types";
import { sampleText } from "./lib/constants";

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
    kokoroVoice: "af_heart",
    kokoroVoiceIds: [],
    audioPath: "",
    audioAutoplayPending: false,
    step: "idle",
    status: "Ready to generate.",
    busy: false,
    error: "",
  });

  useEffect(() => {
    let cancelled = false;

    async function boot() {
      try {
        const [qwen, kokoro] = await Promise.all([
          invoke<ModelBundle>("get_model_bundle_for_runtime", { runtime: "qwen" }),
          invoke<ModelBundle>("get_model_bundle_for_runtime", { runtime: "kokoro" }),
        ]);
        if (cancelled) return;

        const preferredRuntime = localStorage.getItem("chirp.runtime");
        const current =
          preferredRuntime === "kokoro" && kokoro.installed
            ? kokoro
            : preferredRuntime === "qwen" && qwen.installed
              ? qwen
              : qwen.installed
                ? qwen
                : kokoro.installed
                  ? kokoro
                  : qwen;
        setBundle(current);
        const isManagingModels = location.pathname === "/onboard" && new URLSearchParams(location.search).get("manage") === "1";
        if (current.installed && ["/", "/onboard"].includes(location.pathname) && !isManagingModels) navigate("/home", { replace: true });
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
