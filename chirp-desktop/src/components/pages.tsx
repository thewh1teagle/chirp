import { convertFileSrc, invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { open } from "@tauri-apps/plugin-dialog";
import { openPath } from "@tauri-apps/plugin-opener";
import { AnimatePresence, motion } from "framer-motion";
import {
  AudioLines,
  Bot,
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
import { Link, useNavigate } from "react-router-dom";
import voiceCatalogJson from "../assets/voices.json";
import { DownloadedVoice, DownloadProgress, ModelBundle, RunnerInfo, StudioState, VoiceCatalog, VoicePreset } from "../types";
import { cn, formatBytes } from "../utils";
import { AppFrame } from "./AppFrame";
import { CreateStatus } from "./CreateStatus";
import { AgentsPanel } from "./agents/AgentsPanel";
import { Button, Brand, Card, ErrorBlock, Eyebrow, Progress } from "./ui";
import { WaveformPlayer } from "./WaveformPlayer";

const voiceCatalog = voiceCatalogJson as VoiceCatalog;
type VoiceFilter = "all" | "male" | "female" | "american" | "british";

const voiceFilters: Array<{ id: VoiceFilter; label: string }> = [
  { id: "all", label: "All" },
  { id: "female", label: "Female ♀" },
  { id: "male", label: "Male ♂" },
  { id: "american", label: "🇺🇸 American" },
  { id: "british", label: "🇬🇧 British" },
];

type PageProps = {
  bundle: ModelBundle | null;
  setBundle: (bundle: ModelBundle) => void;
};

type HomePageProps = PageProps & {
  studio: StudioState;
  setStudio: Dispatch<SetStateAction<StudioState>>;
};

export function OnboardPage({ bundle, setBundle }: PageProps) {
  const navigate = useNavigate();
  const [progress, setProgress] = useState<DownloadProgress | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    const unlisten = listen<DownloadProgress>("model_download_progress", (event) => {
      setProgress(event.payload);
    });
    return () => {
      unlisten.then((off) => off());
    };
  }, []);

  async function downloadModel() {
    setBusy(true);
    setError("");
    try {
      const installed = await invoke<ModelBundle>("download_model_bundle");
      setBundle(installed);
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

  return (
    <main className="grid min-h-screen place-items-center bg-background p-6 text-primary sm:p-12">
      <section className="w-full max-w-[580px]">
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, ease: "easeOut" }}
        >
          <Brand />
          <h1 className="mt-8 text-4xl font-semibold tracking-tight text-primary sm:text-5xl">
            Local voice <br />
            <span className="text-secondary opacity-40 italic">reimagined.</span>
          </h1>
          <p className="mt-6 max-w-[480px] text-lg leading-relaxed text-secondary opacity-70">
            A professional-grade voice engine that runs entirely on your hardware. Total privacy, zero latency.
          </p>

          <Card className="mt-12 overflow-hidden border-none bg-white p-0 shadow-2xl">
            <div className="flex flex-col gap-8 p-8 sm:flex-row sm:items-center sm:justify-between">
              <div className="space-y-1">
                <Eyebrow>Local Infrastructure</Eyebrow>
                <h3 className="text-base font-semibold tracking-tight">{bundle?.version ?? "chirp-v0.1.3-standard"}</h3>
                <p className="text-xs text-secondary opacity-40">Initial setup: ~1.3GB storage</p>
              </div>
              <Button onClick={downloadModel} disabled={busy} className="h-12 px-8 text-sm shadow-lg shadow-primary/5">
                {busy ? (
                  <span className="flex items-center gap-2">
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Preparing...
                  </span>
                ) : (
                  <span className="flex items-center gap-2">
                    <Download className="h-4 w-4" />
                    Install Models
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
  const { text, referencePath, languages, language, audioPath, step, status, busy, error } = studio;
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
            model_path: currentBundle.model_path,
            codec_path: currentBundle.codec_path,
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
    const current = bundle ?? (await invoke<ModelBundle>("get_model_bundle"));
    setBundle(current);
    if (!current.installed) {
      navigate("/onboard", { replace: true });
      return;
    }
    if (!text.trim()) {
      updateStudio({ status: "Input text required." });
      return;
    }

    updateStudio({ busy: true, error: "", audioPath: "" });
    try {
      updateStudio({ step: "starting", status: "Initializing Engine..." });
      await invoke<RunnerInfo>("start_runner");

      updateStudio({ step: "loading", status: "Loading models..." });
      await invoke("load_model", {
        request: {
          model_path: current.model_path,
          codec_path: current.codec_path,
          temperature: 0.9,
          top_k: 50,
        },
      });

      const supportedLanguages = await invoke<string[]>("get_languages");
      updateStudio({ languages: supportedLanguages.length ? supportedLanguages : ["auto"] });
      const selectedLanguage = supportedLanguages.includes(language) ? language : "auto";
      if (selectedLanguage !== language) updateStudio({ language: "auto" });

      updateStudio({ step: "creating", status: "Generating audio..." });
      const output = await invoke<string>("synthesize", {
        request: {
          input: text,
          voice_reference: referencePath || null,
          language: selectedLanguage,
        },
      });
      updateStudio({ audioPath: output, step: "done", status: "Generation complete." });
    } catch (err) {
      updateStudio({ step: "idle", error: String(err), status: "Generation failed." });
    } finally {
      updateStudio({ busy: false });
    }
  }

  return (
    <AppFrame bundle={bundle}>
      <div className="w-full max-w-[1200px] space-y-10">
        <StudioHeader bundle={bundle} />

        <div className="grid gap-8 lg:grid-cols-[1fr_320px]">
          <div className="space-y-8">
            <EditorCard busy={busy} text={text} setText={(nextText) => updateStudio({ text: nextText })} createVoice={createVoice} />

            <AnimatePresence>
              {audioPath && (
                <motion.div 
                  initial={{ opacity: 0, y: 12 }} 
                  animate={{ opacity: 1, y: 0 }} 
                  exit={{ opacity: 0, y: 8 }}
                >
                  <WaveformPlayer src={audioSrc} sourcePath={audioPath} filename={audioPath.split(/[\\/]/).pop() || "generated-audio.wav"} />
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          <aside className="space-y-8">
            <VoiceSettings
              busy={busy}
              language={language}
              languages={languages}
              referencePath={referencePath}
              chooseReference={chooseReference}
              setLanguage={(nextLanguage) => updateStudio({ language: nextLanguage })}
              setReferencePath={(nextPath) => updateStudio({ referencePath: nextPath })}
            />

            <AnimatePresence>
              {busy && (
                <motion.div
                  initial={{ opacity: 0, scale: 0.98 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.98 }}
                >
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
      <section className="w-full max-w-[760px] space-y-10">
        <MainNav active="agents" />

        <header className="space-y-3">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl border border-border/40 bg-white shadow-sm">
              <Bot className="h-5 w-5 text-primary" />
            </div>
            <div>
              <p className="text-[10px] font-black uppercase tracking-[0.24em] text-secondary opacity-35">Local API</p>
              <h1 className="text-3xl font-semibold tracking-tight text-primary sm:text-4xl">Agent Workspace</h1>
            </div>
          </div>
          <p className="max-w-[520px] text-base leading-7 text-secondary opacity-60">
            Start Chirp's local HTTP API, open Swagger, and copy agent-ready instructions for using speech synthesis from tools.
          </p>
        </header>

        <AgentsPanel />
      </section>
    </AppFrame>
  );
}

function StudioHeader({ bundle }: { bundle: ModelBundle | null }) {
  return (
    <header className="space-y-8">
      <MainNav active="studio" />
      <div className="flex flex-col gap-6 sm:flex-row sm:items-end sm:justify-between">
      <div className="space-y-3">
        <div className="flex items-center gap-3">
          <Brand />
          <div className="h-1 w-1 rounded-full bg-border/40" />
          <span className="text-[10px] font-black uppercase tracking-[0.3em] text-secondary opacity-30">Production Studio</span>
        </div>
        <h1 className="text-3xl font-semibold tracking-tight text-primary sm:text-4xl">Speech Studio</h1>
      </div>
      <div className="flex items-center gap-4">
        <div className="hidden text-right sm:block space-y-0.5">
          <p className="text-[9px] font-bold uppercase tracking-widest text-secondary opacity-30">System Status</p>
          <p className="text-sm font-semibold text-primary">{bundle?.version.split("-").pop() ?? "v0.1.3"}</p>
        </div>
        <div className="h-10 w-[1px] bg-border/20" />
        <Link to="/settings" className="flex h-10 w-10 items-center justify-center rounded-full border border-border/60 bg-white text-secondary transition-all hover:border-primary hover:text-primary hover:scale-105 shadow-sm">
          <Settings className="h-4 w-4" />
        </Link>
      </div>
      </div>
    </header>
  );
}

function MainNav({ active }: { active: "studio" | "agents" }) {
  const items = [
    { id: "studio", label: "Studio", to: "/home", icon: AudioLines },
    { id: "agents", label: "Agents", to: "/agents", icon: Bot },
  ] as const;

  return (
    <nav className="inline-flex rounded-2xl border border-border/30 bg-white p-1.5 shadow-sm">
      {items.map((item) => {
        const Icon = item.icon;
        const isActive = active === item.id;
        return (
          <Link
            key={item.id}
            to={item.to}
            className={cn(
              "flex h-10 items-center gap-2 rounded-xl px-4 text-[11px] font-black uppercase tracking-[0.16em] transition-all",
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
        className="min-h-[320px] w-full resize-none bg-white p-8 text-lg leading-relaxed text-primary outline-none placeholder:text-secondary/20 font-medium"
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
  chooseReference,
  setLanguage,
  setReferencePath,
}: {
  busy: boolean;
  language: string;
  languages: string[];
  referencePath: string;
  chooseReference: () => void;
  setLanguage: (language: string) => void;
  setReferencePath: (path: string) => void;
}) {
  const referenceSrc = useMemo(() => (referencePath ? convertFileSrc(referencePath) : ""), [referencePath]);
  const [voiceBusy, setVoiceBusy] = useState("");
  const [voiceError, setVoiceError] = useState("");
  const [libraryOpen, setLibraryOpen] = useState(false);

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
          <Eyebrow className="mb-0">Voice Cloning</Eyebrow>
        </div>
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
      </div>

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
        className="w-full max-w-[720px] overflow-hidden rounded-2xl border border-border/30 bg-white shadow-2xl"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <audio ref={audioRef} onEnded={() => setPreviewVoiceId("")} />
        <div className="space-y-5 border-b border-border/10 p-6">
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

        <div className="max-h-[58vh] overflow-y-auto p-6">
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
