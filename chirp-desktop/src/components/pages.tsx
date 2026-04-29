import { convertFileSrc, invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { open } from "@tauri-apps/plugin-dialog";
import { openPath } from "@tauri-apps/plugin-opener";
import { AnimatePresence, motion } from "framer-motion";
import {
  AudioLines,
  Bot,
  Check,
  ChevronRight,
  Download,
  FileAudio,
  FolderOpen,
  Languages,
  Loader2,
  Pause,
  Play,
  Plus,
  Settings,
  Sparkles,
  UserRound,
  X,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import type { Dispatch, SetStateAction } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import voiceCatalogJson from "../assets/voices.json";
import { DownloadedVoice, DownloadProgress, ModelBundle, ModelSource, ModelSources, RunnerInfo, StudioState, VoiceCatalog, VoicePreset } from "../types";
import { cn, formatBytes } from "../utils";
import { AppFrame } from "./AppFrame";
import { CreateStatus } from "./CreateStatus";
import { AgentsPanel } from "./agents/AgentsPanel";
import { Button, Brand, Card, ErrorBlock, Eyebrow, Progress } from "./ui";
import { WaveformPlayer } from "./WaveformPlayer";

const voiceCatalog = voiceCatalogJson as VoiceCatalog;
type VoiceFilter = "all" | "male" | "female" | "american" | "british";
type KokoroVoiceFilter = "all" | "male" | "female" | "english" | "japanese" | "spanish" | "french" | "hindi" | "italian" | "portuguese";

const voiceFilters: Array<{ id: VoiceFilter; label: string }> = [
  { id: "all", label: "All" },
  { id: "female", label: "Female ♀" },
  { id: "male", label: "Male ♂" },
  { id: "american", label: "🇺🇸 American" },
  { id: "british", label: "🇬🇧 British" },
];

const kokoroVoices = [
  { id: "af_heart", name: "Heart", language: "American English", flag: "🇺🇸", gender: "female", grade: "A" },
  { id: "af_bella", name: "Bella", language: "American English", flag: "🇺🇸", gender: "female", grade: "A-" },
  { id: "af_nicole", name: "Nicole", language: "American English", flag: "🇺🇸", gender: "female", grade: "B-" },
  { id: "af_aoede", name: "Aoede", language: "American English", flag: "🇺🇸", gender: "female", grade: "C+" },
  { id: "af_kore", name: "Kore", language: "American English", flag: "🇺🇸", gender: "female", grade: "C+" },
  { id: "af_sarah", name: "Sarah", language: "American English", flag: "🇺🇸", gender: "female", grade: "C+" },
  { id: "af_nova", name: "Nova", language: "American English", flag: "🇺🇸", gender: "female", grade: "C" },
  { id: "af_sky", name: "Sky", language: "American English", flag: "🇺🇸", gender: "female", grade: "C-" },
  { id: "am_michael", name: "Michael", language: "American English", flag: "🇺🇸", gender: "male", grade: "C+" },
  { id: "am_fenrir", name: "Fenrir", language: "American English", flag: "🇺🇸", gender: "male", grade: "C+" },
  { id: "am_puck", name: "Puck", language: "American English", flag: "🇺🇸", gender: "male", grade: "C+" },
  { id: "bf_emma", name: "Emma", language: "British English", flag: "🇬🇧", gender: "female", grade: "B-" },
  { id: "bf_isabella", name: "Isabella", language: "British English", flag: "🇬🇧", gender: "female", grade: "C" },
  { id: "bf_alice", name: "Alice", language: "British English", flag: "🇬🇧", gender: "female", grade: "D" },
  { id: "bm_george", name: "George", language: "British English", flag: "🇬🇧", gender: "male", grade: "C" },
  { id: "bm_fable", name: "Fable", language: "British English", flag: "🇬🇧", gender: "male", grade: "C" },
  { id: "bm_lewis", name: "Lewis", language: "British English", flag: "🇬🇧", gender: "male", grade: "D+" },
  { id: "jf_alpha", name: "Alpha", language: "Japanese", flag: "🇯🇵", gender: "female", grade: "C+" },
  { id: "jf_gongitsune", name: "Gongitsune", language: "Japanese", flag: "🇯🇵", gender: "female", grade: "C" },
  { id: "jf_nezumi", name: "Nezumi", language: "Japanese", flag: "🇯🇵", gender: "female", grade: "C-" },
  { id: "jf_tebukuro", name: "Tebukuro", language: "Japanese", flag: "🇯🇵", gender: "female", grade: "C" },
  { id: "jm_kumo", name: "Kumo", language: "Japanese", flag: "🇯🇵", gender: "male", grade: "C-" },
  { id: "ef_dora", name: "Dora", language: "Spanish", flag: "🇪🇸", gender: "female" },
  { id: "em_alex", name: "Alex", language: "Spanish", flag: "🇪🇸", gender: "male" },
  { id: "em_santa", name: "Santa", language: "Spanish", flag: "🇪🇸", gender: "male" },
  { id: "ff_siwis", name: "Siwis", language: "French", flag: "🇫🇷", gender: "female", grade: "B-" },
  { id: "hf_alpha", name: "Alpha", language: "Hindi", flag: "🇮🇳", gender: "female", grade: "C" },
  { id: "hf_beta", name: "Beta", language: "Hindi", flag: "🇮🇳", gender: "female", grade: "C" },
  { id: "hm_omega", name: "Omega", language: "Hindi", flag: "🇮🇳", gender: "male", grade: "C" },
  { id: "hm_psi", name: "Psi", language: "Hindi", flag: "🇮🇳", gender: "male", grade: "C" },
  { id: "if_sara", name: "Sara", language: "Italian", flag: "🇮🇹", gender: "female", grade: "C" },
  { id: "im_nicola", name: "Nicola", language: "Italian", flag: "🇮🇹", gender: "male", grade: "C" },
  { id: "pf_dora", name: "Dora", language: "Brazilian Portuguese", flag: "🇧🇷", gender: "female" },
  { id: "pm_alex", name: "Alex", language: "Brazilian Portuguese", flag: "🇧🇷", gender: "male" },
  { id: "pm_santa", name: "Santa", language: "Brazilian Portuguese", flag: "🇧🇷", gender: "male" },
] as const;

const kokoroVoiceFilters: Array<{ id: KokoroVoiceFilter; label: string }> = [
  { id: "all", label: "All" },
  { id: "female", label: "Female ♀" },
  { id: "male", label: "Male ♂" },
  { id: "english", label: "🇺🇸/🇬🇧 English" },
  { id: "japanese", label: "🇯🇵 Japanese" },
  { id: "spanish", label: "🇪🇸 Spanish" },
  { id: "french", label: "🇫🇷 French" },
  { id: "hindi", label: "🇮🇳 Hindi" },
  { id: "italian", label: "🇮🇹 Italian" },
  { id: "portuguese", label: "🇧🇷 Portuguese" },
];

type PageProps = {
  bundle: ModelBundle | null;
  setBundle: (bundle: ModelBundle) => void;
};
type RuntimeId = ModelBundle["runtime"];

type HomePageProps = PageProps & {
  studio: StudioState;
  setStudio: Dispatch<SetStateAction<StudioState>>;
};

export function OnboardPage({ bundle, setBundle }: PageProps) {
  const navigate = useNavigate();
  const location = useLocation();
  const [runtime, setRuntime] = useState<RuntimeId>("kokoro");
  const [bundles, setBundles] = useState<Record<RuntimeId, ModelBundle | null>>({ qwen: null, kokoro: null });
  const [sources, setSources] = useState<ModelSource[]>([]);
  const [progress, setProgress] = useState<DownloadProgress | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const manageMode = new URLSearchParams(location.search).get("manage") === "1";

  useEffect(() => {
    const unlisten = listen<DownloadProgress>("model_download_progress", (event) => {
      setProgress(event.payload);
    });
    return () => {
      unlisten.then((off) => off());
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    async function refreshBundles() {
      try {
        const [qwen, kokoro, modelSources] = await Promise.all([
          invoke<ModelBundle>("get_model_bundle_for_runtime", { runtime: "qwen" }),
          invoke<ModelBundle>("get_model_bundle_for_runtime", { runtime: "kokoro" }),
          invoke<ModelSources>("get_model_sources"),
        ]);
        if (cancelled) return;
        setBundles({ qwen, kokoro });
        setSources(modelSources.runtimes);
        const preferredRuntime = (localStorage.getItem("chirp.runtime") as RuntimeId | null) ?? bundle?.runtime;
        if (preferredRuntime === "kokoro" && kokoro.installed) {
          setRuntime("kokoro");
        } else if (preferredRuntime === "qwen" && qwen.installed) {
          setRuntime("qwen");
        } else if (kokoro.installed) {
          setRuntime("kokoro");
        } else if (qwen.installed) {
          setRuntime("qwen");
        } else {
          setRuntime("kokoro");
        }
      } catch (err) {
        if (!cancelled) setError(String(err));
      }
    }
    refreshBundles();
    return () => {
      cancelled = true;
    };
  }, [bundle?.runtime]);

  async function selectRuntime(nextRuntime: RuntimeId) {
    setBusy(true);
    setError("");
    try {
      setRuntime(nextRuntime);
      const existing = bundles[nextRuntime];
      const selected = existing?.installed ? existing : await invoke<ModelBundle>("download_model_bundle", { runtime: nextRuntime });
      localStorage.setItem("chirp.runtime", selected.runtime);
      setBundles((current) => ({ ...current, [selected.runtime]: selected }));
      await invoke("stop_runner").catch(() => undefined);
      setBundle(selected);
      navigate("/home", { replace: true });
    } catch (err) {
      setError(String(err));
    } finally {
      setBusy(false);
    }
  }

  const progressValue = Math.round((progress?.progress ?? 0) * 100);
  const hasProgressPercent = typeof progress?.progress === "number";
  const progressLabel = hasProgressPercent
    ? `${progressValue}%`
    : progress?.downloaded
      ? formatBytes(progress.downloaded)
      : "Starting";
  const stageLabel = progress?.stage === "extracting" ? "Optimizing models..." : "Downloading voice model...";
  const options = (sources.length ? sources : [
    { id: "kokoro", name: "Kokoro", version: "kokoro-v1.0", recommended: true, size: "~336 MB", description: "Fast multi-voice speech, lighter setup", files: [], directory: "chirp-kokoro-models-kokoro-v1.0" },
    { id: "qwen", name: "Qwen", version: "chirp-models-v0.1.3", recommended: false, size: "~900 MB", description: "Voice clone, multilingual, best on Mac GPU", files: [], directory: "chirp-models-q5_0" },
  ]) as ModelSource[];

  return (
    <main className="grid min-h-screen place-items-center bg-background p-6 text-primary sm:p-12">
      <section className="w-full max-w-[660px]">
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, ease: "easeOut" }}
        >
          <div className="flex items-center justify-between gap-4">
            <Brand />
            {manageMode ? (
              <Link
                to="/settings"
                className="inline-flex h-9 items-center gap-2 rounded-full border border-border/50 bg-white px-3 text-[10px] font-black uppercase tracking-[0.16em] text-secondary shadow-sm transition-all hover:border-primary hover:text-primary"
              >
                <ChevronRight className="h-3.5 w-3.5 rotate-180" />
                Back
              </Link>
            ) : null}
          </div>
          <h1 className="mt-8 text-4xl font-semibold tracking-tight text-primary sm:text-5xl">
            {manageMode ? "Manage voice" : "Local voice"} <br />
            <span className="text-secondary opacity-40 italic">{manageMode ? "models." : "reimagined."}</span>
          </h1>
          <p className="mt-6 max-w-[480px] text-lg leading-relaxed text-secondary opacity-70">
            {manageMode
              ? "Install or switch between local engines. The current runner restarts automatically after a change."
              : "A professional-grade voice engine that runs entirely on your hardware. Total privacy, zero latency."}
          </p>

          <Card className="mt-10 overflow-hidden border-none bg-white p-0 shadow-2xl">
            <div className="grid gap-3 border-b border-border/10 bg-white p-3 sm:grid-cols-2">
              {options.map((option) => {
                const installed = !!bundles[option.id]?.installed;
                const selected = !!bundle?.installed && bundle.runtime === option.id;
                return (
                  <button
                    key={option.id}
                    onClick={() => setRuntime(option.id)}
                    className={cn(
                      "flex min-h-[150px] flex-col justify-between rounded-xl border p-4 text-left transition-all",
                      runtime === option.id ? "border-primary bg-background/50 shadow-sm ring-1 ring-primary" : "border-border/40 bg-white hover:border-secondary/40",
                    )}
                  >
                    <div className="space-y-4">
                      <div className="flex items-start justify-between gap-3">
                        <div>
                          <h3 className="text-base font-semibold tracking-tight">{option.name}</h3>
                          <p className="mt-1 text-[10px] font-black uppercase tracking-[0.18em] text-secondary opacity-35">{option.size}</p>
                        </div>
                        <span
                          className={cn(
                            "inline-flex shrink-0 items-center gap-1 rounded-full px-2 py-1 text-[8px] font-bold uppercase tracking-widest",
                            selected
                              ? "bg-primary text-white"
                              : installed
                                ? "bg-green-50 text-green-700"
                                : "bg-primary text-white",
                          )}
                        >
                          {selected || installed ? <Check className="h-3 w-3" /> : null}
                          {selected ? "Selected" : installed ? "Installed" : option.recommended ? "Recommended" : "Lightweight"}
                        </span>
                      </div>
                      <p className="max-w-[220px] text-xs font-semibold leading-5 text-secondary opacity-55">{option.description}</p>
                    </div>
                    <div className="mt-4 flex items-center gap-2 text-[9px] font-black uppercase tracking-[0.18em] text-secondary opacity-35">
                      <div className={cn("h-1.5 w-1.5 rounded-full", runtime === option.id ? "bg-primary" : "bg-secondary/30")} />
                      Local Runtime
                    </div>
                  </button>
                );
              })}
            </div>
            <div className="flex flex-col gap-5 bg-background/10 p-6 sm:flex-row sm:items-center sm:justify-between">
              <div className="space-y-1">
                <Eyebrow>Local Infrastructure</Eyebrow>
                <h3 className="text-base font-semibold tracking-tight">{bundles[runtime]?.version ?? (runtime === "qwen" ? "chirp-v0.1.3-standard" : "kokoro-v1.0")}</h3>
                <p className="text-xs text-secondary opacity-40">
                  {bundles[runtime]?.installed ? "Already installed locally" : `Initial setup: ${runtime === "qwen" ? "~900MB" : "~336MB"} storage`}
                </p>
              </div>
              <Button onClick={() => selectRuntime(runtime)} disabled={busy || (!!bundle?.installed && bundle.runtime === runtime)} className="h-11 px-6 text-sm shadow-lg shadow-primary/5">
                {busy ? (
                  <span className="flex items-center gap-2">
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Preparing...
                  </span>
                ) : (
                  <span className="flex items-center gap-2">
                    {bundle?.installed && bundle.runtime === runtime ? <Check className="h-4 w-4" /> : bundles[runtime]?.installed ? <Check className="h-4 w-4" /> : <Download className="h-4 w-4" />}
                    {bundle?.installed && bundle.runtime === runtime ? "Selected" : bundles[runtime]?.installed ? "Use Model" : "Install Models"}
                  </span>
                )}
              </Button>
            </div>

            {(busy || progress) && (
              <motion.div 
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="border-t border-border/10 bg-background/30 p-8"
              >
                <div className="mb-4 flex items-center justify-between text-[11px] font-bold uppercase tracking-widest">
                  <span className="flex items-center gap-2 text-primary">
                    {progress?.stage === "extracting" ? <Sparkles className="h-3.5 w-3.5" /> : <Loader2 className="h-3.5 w-3.5 animate-spin" />}
                    {stageLabel}
                  </span>
                  <span className="font-mono text-base">{progressLabel}</span>
                </div>
                <Progress value={progress?.stage === "extracting" ? 100 : hasProgressPercent ? progressValue : 8} />
                {progress?.downloaded ? (
                  <p className="mt-3 text-xs font-medium text-secondary opacity-50">
                    {progress.total ? `${formatBytes(progress.downloaded)} of ${formatBytes(progress.total)}` : `${formatBytes(progress.downloaded)} downloaded`}
                  </p>
                ) : null}
              </motion.div>
            )}
          </Card>
        </motion.div>

        {error && <ErrorBlock className="mt-8">{error}</ErrorBlock>}
      </section>
    </main>
  );
}

export function HomePage({ bundle, setBundle, studio, setStudio }: HomePageProps) {
  const navigate = useNavigate();
  const { text, referencePath, languages, language, kokoroVoice, audioPath, audioAutoplayPending, step, status, busy, error } = studio;
  const loadingLanguagesRef = useRef(false);

  const audioSrc = useMemo(() => (audioPath ? convertFileSrc(audioPath) : ""), [audioPath]);
  const updateStudio = (patch: Partial<StudioState>) => setStudio((current) => ({ ...current, ...patch }));

  useEffect(() => {
    if (!bundle?.installed || busy || languages.length > 1 || loadingLanguagesRef.current) return;
    const currentBundle = bundle;
    loadingLanguagesRef.current = true;

    async function loadLanguages() {
      try {
        await invoke<RunnerInfo>("start_runner");
        await invoke("load_model", {
          request: {
            runtime: currentBundle.runtime,
            model_path: currentBundle.model_path,
            codec_path: currentBundle.codec_path || "",
            voices_path: currentBundle.voices_path || "",
            espeak_data_path: currentBundle.espeak_data_path || "",
            voice: currentBundle.runtime === "kokoro" ? kokoroVoice : "af_heart",
            temperature: 0.9,
            top_k: 50,
          },
        });
        const supportedLanguages = await invoke<string[]>("get_languages");
        if (supportedLanguages.length) updateStudio({ languages: supportedLanguages });
      } catch {
        updateStudio({ languages: ["auto"] });
      } finally {
        loadingLanguagesRef.current = false;
      }
    }

    loadLanguages();
  }, [bundle, busy, languages.length]);

  async function chooseReference() {
    const selected = await open({
      multiple: false,
      filters: [{ name: "WAV audio", extensions: ["wav"] }],
    });
    if (typeof selected === "string") updateStudio({ referencePath: selected });
  }

  async function createVoice() {
    const preferredRuntime = bundle?.runtime ?? localStorage.getItem("chirp.runtime") ?? "qwen";
    const current = bundle ?? (await invoke<ModelBundle>("get_model_bundle_for_runtime", { runtime: preferredRuntime }));
    setBundle(current);
    if (!current.installed) {
      navigate("/onboard", { replace: true });
      return;
    }
    if (!text.trim()) {
      updateStudio({ status: "Input text required." });
      return;
    }

    updateStudio({ busy: true, error: "", audioPath: "", audioAutoplayPending: false });
    try {
      updateStudio({ step: "starting", status: "Initializing Engine..." });
      await invoke<RunnerInfo>("start_runner");

      updateStudio({ step: "loading", status: "Loading models..." });
      await invoke("load_model", {
        request: {
          runtime: current.runtime,
          model_path: current.model_path,
          codec_path: current.codec_path || "",
          voices_path: current.voices_path || "",
          espeak_data_path: current.espeak_data_path || "",
          voice: current.runtime === "kokoro" ? kokoroVoice : "af_heart",
          temperature: 0.9,
          top_k: 50,
        },
      });

      const supportedLanguages = await invoke<string[]>("get_languages");
      updateStudio({ languages: supportedLanguages.length ? supportedLanguages : ["auto"] });
      const selectedLanguage = current.runtime === "kokoro" ? "auto" : supportedLanguages.includes(language) ? language : "auto";
      if (selectedLanguage !== language) updateStudio({ language: "auto" });

      updateStudio({ step: "creating", status: "Generating audio..." });
      const output = await invoke<string>("synthesize", {
        request: {
          input: text,
          voice_reference: current.runtime === "kokoro" ? null : referencePath || null,
          voice: current.runtime === "kokoro" ? kokoroVoice : null,
          language: selectedLanguage,
        },
      });
      updateStudio({ audioPath: output, audioAutoplayPending: true, step: "done", status: "Generation complete." });
    } catch (err) {
      updateStudio({ step: "idle", error: String(err), status: "Generation failed." });
    } finally {
      updateStudio({ busy: false });
    }
  }

  return (
    <AppFrame bundle={bundle}>
      <div className="w-full max-w-[1200px]">
        <StudioHeader bundle={bundle} />
        <div className="grid gap-12 mt-4 lg:grid-cols-[1fr_360px]">
          <div className="space-y-6">
            <EditorCard
              busy={busy}
              text={text}
              setText={(nextText) => updateStudio({ text: nextText })}
              createVoice={createVoice}
            />

            <AnimatePresence>
              {audioPath && (
                <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 8 }}>
                  <WaveformPlayer
                    src={audioSrc}
                    sourcePath={audioPath}
                    filename={audioPath.split(/[\\/]/).pop() || "generated-audio.wav"}
                    autoPlayOnce={audioAutoplayPending}
                    onAutoPlayConsumed={() => updateStudio({ audioAutoplayPending: false })}
                  />
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          <aside className="space-y-6">
            <VoiceSettings
              busy={busy}
              language={language}
              languages={languages}
              referencePath={referencePath}
              runtime={bundle?.runtime ?? "qwen"}
              kokoroVoice={kokoroVoice}
              chooseReference={chooseReference}
              setLanguage={(nextLanguage) => updateStudio({ language: nextLanguage })}
              setKokoroVoice={(nextVoice) => updateStudio({ kokoroVoice: nextVoice })}
              setReferencePath={(nextPath) => updateStudio({ referencePath: nextPath })}
            />

            <AnimatePresence>
              {busy && (
                <motion.div initial={{ opacity: 0, scale: 0.98 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 0.98 }}>
                  <CreateStatus step={step} status={status} />
                </motion.div>
              )}
            </AnimatePresence>
            {error && <ErrorBlock className="mt-0">{error}</ErrorBlock>}
          </aside>
        </div>
      </div>
    </AppFrame>
  );
}

export function SettingsPage({ bundle }: { bundle: ModelBundle | null }) {
  const [error, setError] = useState("");

  async function openModelsFolder() {
    setError("");
    try {
      const current = bundle ?? (await invoke<ModelBundle>("get_model_bundle"));
      await openPath(current.model_dir);
    } catch (err) {
      setError(String(err));
    }
  }

  return (
    <AppFrame bundle={bundle}>
      <section className="w-full max-w-[640px] space-y-12">
        <header className="space-y-6">
          <Link to="/home" className="inline-flex items-center gap-2 text-[10px] font-black uppercase tracking-[0.2em] text-secondary opacity-40 transition-all hover:text-primary hover:opacity-100">
            <ChevronRight className="h-3 w-3 rotate-180" />
            Studio
          </Link>
          <div className="space-y-1">
            <h1 className="text-3xl font-semibold tracking-tight text-primary sm:text-4xl">System Settings</h1>
            <p className="max-w-[440px] text-base text-secondary opacity-60">Manage local models and high-fidelity voice assets.</p>
          </div>
        </header>

        <div className="space-y-8">
            <div className="space-y-4">
              <h3 className="text-[10px] font-bold uppercase tracking-[0.2em] text-secondary opacity-30">Infrastructure & Storage</h3>
              <Card className="overflow-hidden border-none shadow-xl">
                <div className="flex flex-col gap-8 p-8 sm:flex-row sm:items-center sm:justify-between">
                  <div className="min-w-0 flex-1 space-y-1">
                    <p className="text-[9px] font-bold uppercase tracking-widest text-secondary opacity-30">Models Directory</p>
                    <p className="font-mono text-[11px] text-secondary/70 bg-background/50 px-3 py-2 rounded-lg border border-border/10 truncate">
                      {bundle?.model_dir ?? "Resolving system path..."}
                    </p>
                  </div>
                  <Button variant="outline" onClick={openModelsFolder} className="gap-2 h-10 px-4 shrink-0 text-xs">
                    <FolderOpen className="h-4 w-4" />
                    Open Models Folder
                  </Button>
                </div>

                <div className="flex flex-col gap-8 p-8 sm:flex-row sm:items-center sm:justify-between bg-background/10">
                  <div className="space-y-0.5">
                    <p className="text-[9px] font-bold uppercase tracking-widest text-secondary opacity-30">Engine Specification</p>
                    <p className="text-xl font-semibold tracking-tight text-primary">{bundle?.version ?? "v0.1.3-standard"}</p>
                  </div>
                  <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-white border border-border/40 text-[9px] font-black uppercase tracking-[0.2em] text-green-600 shadow-sm">
                    <div className="h-1.5 w-1.5 rounded-full bg-green-500 animate-pulse" />
                    Live System
                  </div>
                </div>
                <div className="flex flex-col gap-4 border-t border-border/10 p-8 sm:flex-row sm:items-center sm:justify-between">
                  <div className="space-y-1">
                    <p className="text-[9px] font-bold uppercase tracking-widest text-secondary opacity-30">Runtime</p>
                    <p className="text-sm font-semibold tracking-tight text-primary">{bundle?.runtime === "kokoro" ? "Kokoro" : "Qwen"}</p>
                  </div>
                  <Link
                    to="/onboard?manage=1"
                    className="inline-flex h-10 shrink-0 items-center justify-center gap-2 rounded-lg border border-border/80 bg-white px-4 text-xs font-semibold text-primary shadow-sm transition-all hover:border-primary"
                  >
                    Change Model
                    <ChevronRight className="h-3.5 w-3.5" />
                  </Link>
                </div>
              </Card>
            </div>
            {error && <ErrorBlock>{error}</ErrorBlock>}
        </div>
      </section>
    </AppFrame>
  );
}

export function AgentsPage({ bundle }: { bundle: ModelBundle | null }) {
  return (
    <AppFrame bundle={bundle}>
      <section className="w-full max-w-[1200px]">
        <WorkspaceHeader bundle={bundle} active="api" />

        <div className="mx-auto mt-4 max-w-[760px] space-y-10">
          <header className="space-y-3">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-xl border border-border/40 bg-white shadow-sm">
                <Bot className="h-5 w-5 text-primary" />
              </div>
              <div>
                <p className="text-[10px] font-black uppercase tracking-[0.24em] text-secondary opacity-35">Local Service</p>
                <h1 className="text-3xl font-semibold tracking-tight text-primary sm:text-4xl">API Workspace</h1>
              </div>
            </div>
            <p className="max-w-[520px] text-base leading-7 text-secondary opacity-60">
              Start Chirp's local HTTP API, open Swagger, and copy agent-ready instructions for using speech synthesis from tools.
            </p>
          </header>

          <AgentsPanel />
        </div>
      </section>
    </AppFrame>
  );
}

function StudioHeader({ bundle }: { bundle: ModelBundle | null }) {
  return (
    <WorkspaceHeader bundle={bundle} active="studio" />
  );
}

function WorkspaceHeader({ bundle, active }: { bundle: ModelBundle | null; active: "studio" | "api" }) {
  return (
    <header className="flex h-[88px] items-center justify-between">
      <div className="flex items-center gap-6">
        <Brand />
        <MainNav active={active} />
      </div>
      <div className="flex items-center gap-4">
        <span className="text-[10px] font-medium tracking-widest text-secondary opacity-40 uppercase">
          {bundle?.version.split("-").pop() ?? "v0.1.3"}
        </span>
        <Link to="/settings" className="flex h-8 w-8 items-center justify-center rounded-full border border-border/60 text-secondary transition-all hover:border-primary hover:text-primary">
          <Settings className="h-4 w-4" />
        </Link>
      </div>
    </header>
  );
}

function MainNav({ active }: { active: "studio" | "api" }) {
  const items = [
    { id: "studio", label: "Studio", to: "/home", icon: AudioLines },
    { id: "api", label: "API", to: "/agents", icon: Bot },
  ] as const;

  return (
    <nav className="inline-flex min-w-[220px] rounded-2xl border border-border/30 bg-white p-1.5 shadow-sm">
      {items.map((item) => {
        const Icon = item.icon;
        const isActive = active === item.id;
        return (
          <Link
            key={item.id}
            to={item.to}
            className={cn(
              "flex h-10 flex-1 items-center justify-center gap-2 rounded-xl px-4 text-[11px] font-black uppercase tracking-[0.16em] transition-all",
              isActive ? "bg-primary text-white shadow-sm" : "text-secondary opacity-55 hover:text-primary hover:opacity-100",
            )}
          >
            <Icon className="h-4 w-4" />
            {item.label}
          </Link>
        );
      })}
    </nav>
  );
}

function ReferencePlayer({ src }: { src: string }) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);

  const togglePlay = (event: React.MouseEvent) => {
    event.stopPropagation();
    if (!audioRef.current) return;
    if (isPlaying) audioRef.current.pause();
    else audioRef.current.play();
    setIsPlaying(!isPlaying);
  };

  return (
    <div className="flex items-center gap-2">
      <audio ref={audioRef} src={src} onEnded={() => setIsPlaying(false)} />
      <button
        onClick={togglePlay}
        className="flex h-9 w-9 items-center justify-center rounded-full bg-primary text-white transition-all hover:scale-110 active:scale-90 shadow-lg shadow-primary/10 cursor-pointer"
      >
        {isPlaying ? <Pause className="h-4 w-4 fill-current" /> : <Play className="h-4 w-4 fill-current ml-0.5" />}
      </button>
    </div>
  );
}

function EditorCard({
  busy,
  text,
  setText,
  createVoice,
}: {
  busy: boolean;
  text: string;
  setText: (text: string) => void;
  createVoice: () => void;
}) {
  return (
    <Card className="relative overflow-hidden p-0 shadow-xl border-none">
      <textarea
        id="text"
        value={text}
        placeholder="Paste your script here..."
        onChange={(event) => setText(event.currentTarget.value)}
        disabled={busy}
        className="min-h-[320px] w-full resize-none bg-white p-8 text-left text-lg font-medium leading-relaxed text-primary outline-none placeholder:text-secondary/20"
      />
      <div className="flex items-center justify-between border-t border-border/10 bg-background/10 px-8 py-5">
        <div className="flex items-center gap-4 text-[10px] font-bold uppercase tracking-[0.2em] text-secondary opacity-40">
          <span className={cn("transition-colors", text.length > 500 ? "text-amber-600 opacity-100" : "")}>{text.length} Characters</span>
        </div>
        <Button onClick={createVoice} disabled={busy || !text.trim()} className="h-12 px-8 text-sm shadow-xl shadow-primary/5 transition-transform hover:scale-[1.01]">
          {busy ? (
            <span className="flex items-center gap-2">
              <Loader2 className="h-4 w-4 animate-spin" />
                  Generating...
            </span>
          ) : (
            <span className="flex items-center gap-2">
              <Play className="h-3.5 w-3.5 fill-current" />
              Generate
            </span>
          )}
        </Button>
      </div>
    </Card>
  );
}

function VoiceSettings({
  busy,
  language,
  languages,
  referencePath,
  runtime,
  kokoroVoice,
  chooseReference,
  setLanguage,
  setKokoroVoice,
  setReferencePath,
}: {
  busy: boolean;
  language: string;
  languages: string[];
  referencePath: string;
  runtime: ModelBundle["runtime"];
  kokoroVoice: string;
  chooseReference: () => void;
  setLanguage: (language: string) => void;
  setKokoroVoice: (voice: string) => void;
  setReferencePath: (path: string) => void;
}) {
  const referenceSrc = useMemo(() => (referencePath ? convertFileSrc(referencePath) : ""), [referencePath]);
  const [voiceBusy, setVoiceBusy] = useState("");
  const [voiceError, setVoiceError] = useState("");
  const [libraryOpen, setLibraryOpen] = useState(false);
  const selectedKokoroVoice = kokoroVoices.find((voice) => voice.id === kokoroVoice) ?? kokoroVoices[0];

  async function choosePresetVoice(voice: VoicePreset) {
    setVoiceBusy(voice.id);
    setVoiceError("");
    try {
      const downloaded = await invoke<DownloadedVoice>("download_voice", {
        request: {
          id: voice.id,
          url: voice.url,
        },
      });
      setReferencePath(downloaded.path);
    } catch (err) {
      setVoiceError(String(err));
    } finally {
      setVoiceBusy("");
    }
  }

  return (
    <Card className="space-y-8 p-6 border-none shadow-xl">
      <div className="space-y-5">
        <div className="flex items-center gap-2.5">
          <AudioLines className="h-4 w-4 text-secondary opacity-40" />
          <Eyebrow className="mb-0">{runtime === "kokoro" ? "Voice Preset" : "Voice Cloning"}</Eyebrow>
        </div>
        {runtime === "kokoro" ? (
          <div className="rounded-xl border border-border/30 bg-white p-5 shadow-sm">
            <div className="flex items-center gap-3">
              <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-background text-lg">{selectedKokoroVoice.flag}</div>
              <div className="min-w-0">
                <p className="text-xs font-bold tracking-tight text-primary">{selectedKokoroVoice.name}</p>
                <p className="text-[10px] font-bold uppercase tracking-widest text-secondary opacity-35">{selectedKokoroVoice.language}</p>
              </div>
              <Button
                variant="ghost"
                onClick={() => setLibraryOpen(true)}
                disabled={busy}
                className="ml-auto h-8 shrink-0 gap-1.5 rounded-full border border-border/40 bg-white px-3 text-[10px] font-black uppercase tracking-[0.16em] shadow-sm hover:border-primary/30"
              >
                Change
                <ChevronRight className="h-3.5 w-3.5" />
              </Button>
            </div>
          </div>
        ) : (
          <div
            className={cn(
              "group relative flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed p-6 transition-all hover:bg-background/40",
              referencePath ? "border-primary bg-background/20" : "border-border/60 hover:border-secondary/30",
            )}
            onClick={chooseReference}
          >
            {referencePath ? (
              <div className="text-center w-full space-y-3">
                <div className="flex justify-between items-center px-1">
                  <div className="invisible h-9 w-9" />
                  <div className="flex-1 flex flex-col items-center">
                    <div className="h-12 w-12 bg-white rounded-xl shadow-lg flex items-center justify-center mb-3 border border-border/10">
                      <FileAudio className="h-6 w-6 text-primary" />
                    </div>
                    <p className="max-w-[160px] truncate text-xs font-bold tracking-tight text-primary">{referencePath.split(/[\\/]/).pop()}</p>
                  </div>
                  {referenceSrc && <ReferencePlayer src={referenceSrc} />}
                </div>
                <button
                  onClick={(event) => {
                    event.stopPropagation();
                    setReferencePath("");
                  }}
                  className="text-[9px] font-black uppercase tracking-[0.2em] text-secondary opacity-30 hover:text-primary transition-all cursor-pointer"
                >
                  Clear
                </button>
              </div>
            ) : (
              <div className="text-center space-y-3">
                <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-xl bg-white shadow-lg transition-transform group-hover:scale-105 border border-border/10">
                  <Plus className="h-5 w-5 text-secondary opacity-30" />
                </div>
                <div className="space-y-0.5">
                  <p className="text-xs font-bold tracking-tight text-primary">Upload WAV</p>
                  <p className="text-[9px] font-bold uppercase tracking-widest text-secondary opacity-30">Clone any voice</p>
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {runtime === "qwen" ? (
        <>
          <div className="h-[1px] bg-border/10" />

          <div className="rounded-xl border border-border/30 bg-white p-4 shadow-sm">
            <div className="flex items-center justify-between gap-4">
              <div className="min-w-0 space-y-1">
                <div className="flex items-center gap-2">
                  <UserRound className="h-4 w-4 text-secondary opacity-40" />
                  <p className="text-xs font-bold tracking-tight text-primary">Preset Voices</p>
                </div>
                <p className="truncate text-[10px] font-bold uppercase tracking-widest text-secondary opacity-30">
                  {voiceCatalog.voices.length} downloadable references
                </p>
              </div>
              <Button
                variant="ghost"
                onClick={() => setLibraryOpen(true)}
                disabled={busy}
                className="h-8 shrink-0 gap-1.5 rounded-full border border-border/40 bg-white px-3 text-[10px] font-black uppercase tracking-[0.16em] shadow-sm hover:border-primary/30"
              >
                Select
                <ChevronRight className="h-3.5 w-3.5" />
              </Button>
            </div>
            {voiceError ? <p className="text-xs font-medium text-red-900">{voiceError}</p> : null}
          </div>

          <VoiceLibraryDialog
            busy={busy}
            open={libraryOpen}
            voiceBusy={voiceBusy}
            onClose={() => setLibraryOpen(false)}
            onChoose={choosePresetVoice}
          />
        </>
      ) : null}

      {runtime === "kokoro" ? (
        <KokoroVoiceDialog
          busy={busy}
          open={libraryOpen}
          selectedVoice={kokoroVoice}
          onClose={() => setLibraryOpen(false)}
          onChoose={(voice) => {
            setKokoroVoice(voice);
            setLibraryOpen(false);
          }}
        />
      ) : (
        <>
          <div className="h-[1px] bg-border/10" />

          <div className="space-y-5">
            <div className="flex items-center gap-2.5">
              <Languages className="h-4 w-4 text-secondary opacity-40" />
              <Eyebrow className="mb-0">Language</Eyebrow>
            </div>
            <div className="relative group">
              <select
                id="language"
                value={language}
                onChange={(event) => setLanguage(event.currentTarget.value)}
                disabled={busy}
                className="h-12 w-full appearance-none rounded-xl border border-border/30 bg-white px-4 text-xs font-bold tracking-tight text-primary outline-none transition-all focus:border-primary focus:ring-4 focus:ring-primary/5 cursor-pointer shadow-sm group-hover:shadow-md"
              >
                {languages.map((item) => (
                  <option key={item} value={item}>
                    {item === "auto" ? "Detect Automatically" : item[0].toUpperCase() + item.slice(1)}
                  </option>
                ))}
              </select>
              <div className="pointer-events-none absolute right-4 top-1/2 -translate-y-1/2 opacity-20 group-hover:opacity-60 transition-opacity">
                <ChevronRight className="h-3.5 w-3.5 rotate-90" />
              </div>
            </div>
          </div>
        </>
      )}
    </Card>
  );
}

function VoiceLibraryDialog({
  busy,
  open,
  voiceBusy,
  onClose,
  onChoose,
}: {
  busy: boolean;
  open: boolean;
  voiceBusy: string;
  onClose: () => void;
  onChoose: (voice: VoicePreset) => void;
}) {
  const [filter, setFilter] = useState<VoiceFilter>("all");
  const audioRef = useRef<HTMLAudioElement>(null);
  const [previewVoiceId, setPreviewVoiceId] = useState("");
  const visibleVoices = useMemo(
    () =>
      voiceCatalog.voices.filter((voice) => {
        if (filter === "all") return true;
        if (filter === "american") return voice.id.startsWith("american");
        if (filter === "british") return voice.id.startsWith("british");
        if (filter === "male") return voice.id.includes("_m_");
        return voice.id.includes("_f_");
      }),
    [filter],
  );

  useEffect(() => {
    if (!open) return;
    function closeOnEscape(event: KeyboardEvent) {
      if (event.key === "Escape") onClose();
    }
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [open, onClose]);

  function previewVoice(voice: VoicePreset) {
    const audio = audioRef.current;
    if (!audio) return;
    if (previewVoiceId === voice.id && !audio.paused) {
      audio.pause();
      setPreviewVoiceId("");
      return;
    }
    audio.src = voice.url;
    audio.play();
    setPreviewVoiceId(voice.id);
  }

  async function chooseVoice(voice: VoicePreset) {
    await onChoose(voice);
    onClose();
  }

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-primary/20 p-4 backdrop-blur-sm" onMouseDown={onClose}>
      <div
        className="flex h-[min(760px,calc(100vh-32px))] w-full max-w-[720px] flex-col overflow-hidden rounded-2xl border border-border/30 bg-white shadow-2xl"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <audio ref={audioRef} onEnded={() => setPreviewVoiceId("")} />
        <div className="shrink-0 space-y-5 border-b border-border/10 p-6">
          <div className="flex items-start justify-between gap-5">
            <div className="space-y-1">
              <div className="flex items-center gap-2">
                <UserRound className="h-4 w-4 text-secondary opacity-40" />
                <p className="text-xl font-semibold tracking-tight text-primary">Voice Library</p>
              </div>
              <p className="max-w-[460px] text-sm leading-6 text-secondary opacity-60">
                Preset reference voices are downloaded only when selected.
              </p>
            </div>
            <button
              type="button"
              onClick={onClose}
              className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full border border-border/40 bg-white text-secondary shadow-sm transition-all hover:border-primary hover:text-primary"
              aria-label="Close voice library"
            >
              <X className="h-4 w-4" />
            </button>
          </div>

          <div className="flex flex-wrap gap-2">
            {voiceFilters.map((item) => (
              <button
                key={item.id}
                type="button"
                onClick={() => setFilter(item.id)}
                className={cn(
                  "h-8 rounded-full border px-3 text-[10px] font-black uppercase tracking-[0.16em] transition-all",
                  filter === item.id
                    ? "border-primary bg-primary text-white shadow-sm"
                    : "border-border/40 bg-white text-secondary opacity-60 hover:border-primary/30 hover:text-primary hover:opacity-100",
                )}
              >
                {item.label}
              </button>
            ))}
          </div>
        </div>

        <div className="flex-1 overflow-y-auto p-6">
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {visibleVoices.map((voice) => (
              <div
                key={voice.id}
                className={cn(
                  "group min-w-0 rounded-xl border border-border/30 bg-white p-4 text-left shadow-sm transition-all hover:border-primary/30 hover:shadow-md",
                  (busy || !!voiceBusy) && "opacity-60",
                )}
              >
                <div className="flex items-center justify-between gap-2">
                  <p className="truncate text-sm font-bold tracking-tight text-primary">{voice.name}</p>
                  <div className="flex shrink-0 items-center gap-1.5">
                    <button
                      type="button"
                      onClick={() => previewVoice(voice)}
                      disabled={busy || !!voiceBusy}
                      className="flex h-7 w-7 items-center justify-center rounded-full border border-border/40 bg-white text-secondary shadow-sm transition-all hover:border-primary hover:text-primary disabled:cursor-not-allowed disabled:opacity-50"
                      aria-label={`Preview ${voice.name}`}
                    >
                      {previewVoiceId === voice.id ? (
                        <Pause className="h-3.5 w-3.5 fill-current" />
                      ) : (
                        <Play className="h-3.5 w-3.5 fill-current" />
                      )}
                    </button>
                    <button
                      type="button"
                      disabled={busy || !!voiceBusy}
                      onClick={() => chooseVoice(voice)}
                      className="flex h-8 w-8 items-center justify-center rounded-full border border-primary/20 bg-background text-primary shadow-sm transition-all hover:border-primary hover:bg-primary hover:text-white disabled:cursor-not-allowed disabled:opacity-50"
                      aria-label={`Use ${voice.name}`}
                      title={`Use ${voice.name}`}
                    >
                      {voiceBusy === voice.id ? (
                        <Loader2 className="h-4 w-4 animate-spin" />
                      ) : (
                        <Download className="h-4 w-4" />
                      )}
                    </button>
                  </div>
                </div>
                <p className="mt-1 truncate text-[9px] font-bold uppercase tracking-widest text-secondary opacity-35">
                  {voice.id.startsWith("british") ? "🇬🇧 British" : "🇺🇸 American"} / {voice.id.includes("_m_") ? "Male ♂" : "Female ♀"}
                </p>
                <p className="mt-3 line-clamp-2 text-xs leading-5 text-secondary opacity-50">{voice.description}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function KokoroVoiceDialog({
  busy,
  open,
  selectedVoice,
  onClose,
  onChoose,
}: {
  busy: boolean;
  open: boolean;
  selectedVoice: string;
  onClose: () => void;
  onChoose: (voice: string) => void;
}) {
  const [filter, setFilter] = useState<KokoroVoiceFilter>("all");
  const visibleVoices = useMemo(
    () =>
      kokoroVoices.filter((voice) => {
        if (filter === "all") return true;
        if (filter === "male" || filter === "female") return voice.gender === filter;
        if (filter === "english") return voice.language.includes("English");
        if (filter === "portuguese") return voice.language.includes("Portuguese");
        return voice.language.toLowerCase().includes(filter);
      }),
    [filter],
  );

  useEffect(() => {
    if (!open) return;
    function closeOnEscape(event: KeyboardEvent) {
      if (event.key === "Escape") onClose();
    }
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-primary/20 p-4 backdrop-blur-sm" onMouseDown={onClose}>
      <div
        className="flex h-[min(760px,calc(100vh-32px))] w-full max-w-[720px] flex-col overflow-hidden rounded-2xl border border-border/30 bg-white shadow-2xl"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <div className="shrink-0 space-y-5 border-b border-border/10 p-6">
          <div className="flex items-start justify-between gap-5">
            <div className="space-y-1">
              <p className="text-[10px] font-black uppercase tracking-[0.24em] text-secondary opacity-35">Kokoro Voices</p>
              <h3 className="text-2xl font-semibold tracking-tight text-primary">Choose voice</h3>
            </div>
            <button
              type="button"
              onClick={onClose}
              className="flex h-9 w-9 items-center justify-center rounded-full border border-border/40 text-secondary opacity-60 transition-all hover:opacity-100"
              aria-label="Close voice library"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
          <div className="flex flex-wrap gap-2">
            {kokoroVoiceFilters.map((item) => (
              <button
                key={item.id}
                onClick={() => setFilter(item.id)}
                className={cn(
                  "rounded-full border px-3 py-1.5 text-[10px] font-black uppercase tracking-[0.14em] transition-all",
                  filter === item.id ? "border-primary bg-primary text-white" : "border-border/40 bg-white text-secondary opacity-65 hover:opacity-100",
                )}
              >
                {item.label}
              </button>
            ))}
          </div>
        </div>
        <div className="grid flex-1 content-start gap-3 overflow-y-auto p-4 sm:grid-cols-2">
          {visibleVoices.map((voice) => {
            const selected = voice.id === selectedVoice;
            return (
              <button
                key={voice.id}
                disabled={busy}
                onClick={() => onChoose(voice.id)}
                className={cn(
                  "rounded-xl border p-4 text-left transition-all hover:border-primary/30 hover:shadow-sm disabled:opacity-60",
                  selected ? "border-primary bg-background/60" : "border-border/30 bg-white",
                )}
              >
                <div className="flex items-start gap-3">
                  <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-background text-lg">{voice.flag}</div>
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center justify-between gap-2">
                      <p className="truncate text-sm font-bold tracking-tight text-primary">{voice.name}</p>
                      <span className="text-[10px] font-bold text-secondary opacity-45">{voice.gender === "male" ? "♂" : "♀"}</span>
                    </div>
                    <p className="mt-1 text-[10px] font-black uppercase tracking-[0.16em] text-secondary opacity-35">{voice.language}</p>
                    <p className="mt-2 font-mono text-[10px] text-secondary opacity-50">{voice.id}</p>
                  </div>
                  {"grade" in voice ? (
                    <span className="rounded-full border border-border/40 px-2 py-1 text-[9px] font-black uppercase tracking-widest text-secondary opacity-50">
                      {voice.grade}
                    </span>
                  ) : null}
                </div>
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}
