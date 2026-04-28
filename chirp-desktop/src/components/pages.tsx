import { convertFileSrc, invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { open } from "@tauri-apps/plugin-dialog";
import { openPath } from "@tauri-apps/plugin-opener";
import { AnimatePresence, motion } from "framer-motion";
import {
  AudioLines,
  ChevronRight,
  Download,
  FileAudio,
  FolderOpen,
  Languages,
  Loader2,
  Play,
  Plus,
  Settings,
  Sparkles,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { Link, useNavigate } from "react-router";
import { CreateStep, DownloadProgress, ModelBundle, RunnerInfo } from "../types";
import { cn, formatBytes, sampleText } from "../utils";
import { AppFrame } from "./AppFrame";
import { CreateStatus } from "./CreateStatus";
import { Button, Brand, Card, ErrorBlock, Eyebrow, Progress } from "./ui";
import { WaveformPlayer } from "./WaveformPlayer";

type PageProps = {
  bundle: ModelBundle | null;
  setBundle: (bundle: ModelBundle) => void;
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
  const stageLabel = progress?.stage === "extracting" ? "Finishing setup..." : "Downloading voice model...";

  return (
    <main className="grid min-h-screen place-items-center bg-background p-6 text-primary sm:p-12">
      <section className="w-full max-w-[640px]">
        <Brand />
        <h1 className="mt-8 text-4xl font-semibold tracking-tight text-primary sm:text-6xl">
          Local voice <br />
          <span className="text-secondary">reimagined.</span>
        </h1>
        <p className="mt-6 max-w-[540px] text-lg leading-relaxed text-secondary sm:text-xl">
          Chirp brings professional speech synthesis to your device. The model stays local, ensuring speed and total privacy.
        </p>

        <Card className="mt-12 overflow-hidden border-none bg-surface shadow-2xl">
          <div className="flex flex-col gap-6 p-8 sm:flex-row sm:items-center sm:justify-between">
            <div className="space-y-1">
              <Eyebrow>One-time setup</Eyebrow>
              <h3 className="text-lg font-semibold">{bundle?.version ?? "chirp-models-v0.1.3"}</h3>
              <p className="text-sm text-secondary opacity-80">Requires ~1.3GB of disk space</p>
            </div>
            <Button onClick={downloadModel} disabled={busy} className="h-14 px-8 text-base">
              {busy ? (
                <span className="flex items-center gap-2">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Downloading
                </span>
              ) : (
                <span className="flex items-center gap-2">
                  <Download className="h-4 w-4" />
                  Install Model
                </span>
              )}
            </Button>
          </div>

          {(busy || progress) && (
            <div className="border-t border-border/40 bg-background/50 p-8">
              <div className="mb-4 flex items-center justify-between text-sm font-semibold">
                <span className="flex items-center gap-2">
                  {progress?.stage === "extracting" ? <Sparkles className="h-4 w-4" /> : <Download className="h-4 w-4" />}
                  {stageLabel}
                </span>
                <span className="font-mono">{progressValue}%</span>
              </div>
              <Progress value={progress?.stage === "extracting" ? 100 : progressValue} />
              <div className="mt-4 flex items-center justify-between text-xs font-medium text-secondary">
                <span>{progress?.total ? `${formatBytes(progress.downloaded)} of ${formatBytes(progress.total)}` : "Calculating..."}</span>
                {progress?.stage === "downloading" && <span className="animate-pulse">Active Download</span>}
              </div>
            </div>
          )}
        </Card>

        {error && <ErrorBlock className="mt-6">{error}</ErrorBlock>}
      </section>
    </main>
  );
}

export function HomePage({ bundle, setBundle }: PageProps) {
  const navigate = useNavigate();
  const [text, setText] = useState(sampleText);
  const [referencePath, setReferencePath] = useState("");
  const [languages, setLanguages] = useState<string[]>(["auto"]);
  const [language, setLanguage] = useState("auto");
  const [audioPath, setAudioPath] = useState("");
  const [step, setStep] = useState<CreateStep>("idle");
  const [status, setStatus] = useState("Ready to create.");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const audioSrc = useMemo(() => (audioPath ? convertFileSrc(audioPath) : ""), [audioPath]);

  async function chooseReference() {
    const selected = await open({
      multiple: false,
      filters: [{ name: "WAV audio", extensions: ["wav"] }],
    });
    if (typeof selected === "string") setReferencePath(selected);
  }

  async function createVoice() {
    const current = bundle ?? (await invoke<ModelBundle>("get_model_bundle"));
    setBundle(current);
    if (!current.installed) {
      navigate("/onboard", { replace: true });
      return;
    }
    if (!text.trim()) {
      setStatus("Add text before creating audio.");
      return;
    }

    setBusy(true);
    setError("");
    setAudioPath("");
    try {
      setStep("starting");
      setStatus("Initializing...");
      await invoke<RunnerInfo>("start_runner");

      setStep("loading");
      setStatus("Loading weights...");
      await invoke("load_model", {
        request: {
          model_path: current.model_path,
          codec_path: current.codec_path,
          temperature: 0.9,
          top_k: 50,
        },
      });

      const supportedLanguages = await invoke<string[]>("get_languages");
      setLanguages(supportedLanguages.length ? supportedLanguages : ["auto"]);
      const selectedLanguage = supportedLanguages.includes(language) ? language : "auto";
      if (selectedLanguage !== language) setLanguage("auto");

      setStep("creating");
      setStatus("Generating speech...");
      const output = await invoke<string>("synthesize", {
        request: {
          input: text,
          voice_reference: referencePath || null,
          language: selectedLanguage,
        },
      });
      setAudioPath(output);
      setStep("done");
      setStatus("Generation complete.");
    } catch (err) {
      setStep("idle");
      setError(String(err));
      setStatus("Failed to generate.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <AppFrame bundle={bundle}>
      <div className="w-full max-w-[1040px] space-y-8">
        <StudioHeader bundle={bundle} />

        <div className="grid gap-6 lg:grid-cols-[1fr_320px]">
          <div className="space-y-6">
            <EditorCard busy={busy} text={text} setText={setText} createVoice={createVoice} />

            <AnimatePresence>
              {audioPath && (
                <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 10 }}>
                  <WaveformPlayer src={audioSrc} filename={audioPath.split(/[\\/]/).pop() || "preview.wav"} />
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
              chooseReference={chooseReference}
              setLanguage={setLanguage}
              setReferencePath={setReferencePath}
            />

            {busy && <CreateStatus step={step} status={status} />}
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
      <section className="w-full max-w-[680px] space-y-12">
        <header className="space-y-4">
          <Link to="/home" className="inline-flex items-center gap-2 text-[10px] font-bold uppercase tracking-[0.2em] text-secondary opacity-60 transition-colors hover:text-primary hover:opacity-100">
            <ChevronRight className="h-3 w-3 rotate-180" />
            Studio
          </Link>
          <h1 className="text-4xl font-semibold tracking-tight text-primary sm:text-5xl">Settings</h1>
          <p className="max-w-[480px] text-lg text-secondary opacity-80">Configure your local voice synthesis engine and manage model storage.</p>
        </header>

        <div className="space-y-6">
          <div className="space-y-3">
            <h3 className="text-[10px] font-bold uppercase tracking-[0.2em] text-secondary opacity-40">Model Management</h3>
            <Card className="divide-y divide-border/30 overflow-hidden border-border/60 shadow-md rounded-xl">
              <div className="flex flex-col gap-6 p-6 sm:flex-row sm:items-center sm:justify-between">
                <div className="min-w-0 flex-1 space-y-1">
                  <p className="text-[10px] font-bold uppercase tracking-widest text-secondary opacity-40">Storage Location</p>
                  <p className="font-mono text-[11px] text-secondary/80 truncate">{bundle?.model_dir ?? "Locating model directory..."}</p>
                </div>
                <Button variant="outline" onClick={openModelsFolder} className="gap-2 h-9 px-4 shrink-0 text-xs">
                  <FolderOpen className="h-3.5 w-3.5" />
                  Open Folder
                </Button>
              </div>

              <div className="flex flex-col gap-6 p-6 sm:flex-row sm:items-center sm:justify-between bg-background/30">
                <div className="space-y-0.5">
                  <p className="text-[10px] font-bold uppercase tracking-widest text-secondary opacity-40">Active Model</p>
                  <p className="text-lg font-semibold tracking-tight">{bundle?.version ?? "v0.1.3"}</p>
                </div>
                <div className="inline-flex items-center gap-2 px-2.5 py-1 rounded-md bg-white border border-border/50 text-[9px] font-bold uppercase tracking-[0.15em] text-green-600 shadow-sm">
                  <div className="h-1.5 w-1.5 rounded-full bg-green-500 animate-pulse" />
                  Latest
                </div>
              </div>
            </Card>
          </div>
        </div>

        {error && <ErrorBlock>{error}</ErrorBlock>}
      </section>
    </AppFrame>
  );
}

function StudioHeader({ bundle }: { bundle: ModelBundle | null }) {
  return (
    <header className="flex flex-col gap-6 sm:flex-row sm:items-end sm:justify-between">
      <div className="space-y-3">
        <div className="flex items-center gap-3">
          <Brand className="h-7 min-w-[60px] text-[11px]" />
          <div className="h-1 w-1 rounded-full bg-border" />
          <span className="text-xs font-bold uppercase tracking-widest text-secondary opacity-60">Studio</span>
        </div>
        <h1 className="text-4xl font-semibold tracking-tight text-primary sm:text-5xl">Speech Synthesis</h1>
      </div>
      <div className="flex items-center gap-4">
        <div className="hidden text-right sm:block">
          <p className="text-[10px] font-bold uppercase tracking-wider text-secondary opacity-50">Local Engine</p>
          <p className="text-sm font-medium">{bundle?.version.split("-").pop() ?? "v0.1.3"}</p>
        </div>
        <div className="h-10 w-[1px] bg-border/60" />
        <Link to="/settings" className="flex h-10 w-10 items-center justify-center rounded-full border border-border bg-surface text-secondary transition-colors hover:border-primary hover:text-primary">
          <Settings className="h-5 w-5" />
        </Link>
      </div>
    </header>
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
    <Card className="relative overflow-hidden p-0 shadow-xl">
      <textarea
        id="text"
        value={text}
        placeholder="Enter text to synthesize..."
        onChange={(event) => setText(event.currentTarget.value)}
        disabled={busy}
        className="min-h-[320px] w-full resize-none bg-surface p-8 text-xl leading-relaxed text-primary outline-none placeholder:text-secondary/30"
      />
      <div className="flex items-center justify-between border-t border-border/40 bg-background/30 px-6 py-4">
        <div className="flex items-center gap-4 text-xs font-bold uppercase tracking-widest text-secondary opacity-60">
          <span className={cn(text.length > 500 ? "text-amber-600" : "")}>{text.length} Characters</span>
        </div>
        <Button onClick={createVoice} disabled={busy || !text.trim()} className="h-11 px-6 text-sm">
          {busy ? (
            <span className="flex items-center gap-2">
              <Loader2 className="h-4 w-4 animate-spin" />
              Creating
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
  return (
    <Card className="space-y-6 p-6">
      <div className="space-y-4">
        <div className="flex items-center gap-2">
          <AudioLines className="h-4 w-4 text-secondary" />
          <Eyebrow className="mb-0">Voice Clone</Eyebrow>
        </div>
        <div
          className={cn(
            "group relative flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed p-6 transition-all hover:bg-background",
            referencePath ? "border-primary bg-background" : "border-border hover:border-secondary",
          )}
          onClick={chooseReference}
        >
          {referencePath ? (
            <div className="text-center">
              <FileAudio className="mx-auto mb-2 h-8 w-8 text-primary" />
              <p className="max-w-[180px] truncate text-sm font-semibold">{referencePath.split(/[\\/]/).pop()}</p>
              <button
                onClick={(event) => {
                  event.stopPropagation();
                  setReferencePath("");
                }}
                className="mt-2 text-xs font-bold text-secondary hover:text-primary"
              >
                Remove
              </button>
            </div>
          ) : (
            <div className="text-center">
              <div className="mx-auto mb-2 flex h-10 w-10 items-center justify-center rounded-full bg-background group-hover:bg-surface">
                <Plus className="h-5 w-5 text-secondary" />
              </div>
              <p className="text-sm font-medium text-secondary">Add reference WAV</p>
              <p className="mt-1 text-[10px] text-secondary/50">Clone any voice</p>
            </div>
          )}
        </div>
      </div>

      <div className="h-[1px] bg-border/40" />

      <div className="space-y-4">
        <div className="flex items-center gap-2">
          <Languages className="h-4 w-4 text-secondary" />
          <Eyebrow className="mb-0">Language</Eyebrow>
        </div>
        <div className="relative">
          <select
            id="language"
            value={language}
            onChange={(event) => setLanguage(event.currentTarget.value)}
            disabled={busy}
            className="h-12 w-full appearance-none rounded-xl border border-border bg-surface px-4 text-sm font-medium outline-none transition-all focus:border-primary focus:ring-4 focus:ring-primary/5"
          >
            {languages.map((item) => (
              <option key={item} value={item}>
                {item === "auto" ? "Detect Automatically" : item[0].toUpperCase() + item.slice(1)}
              </option>
            ))}
          </select>
          <div className="pointer-events-none absolute right-4 top-1/2 -translate-y-1/2 opacity-40">
            <ChevronRight className="h-4 w-4 rotate-90" />
          </div>
        </div>
      </div>
    </Card>
  );
}
