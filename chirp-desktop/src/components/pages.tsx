import { convertFileSrc, invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { open } from "@tauri-apps/plugin-dialog";
import { openPath, openUrl } from "@tauri-apps/plugin-opener";
import { AnimatePresence, motion } from "framer-motion";
import {
  AudioLines,
  ChevronRight,
  Check,
  Clipboard,
  Download,
  ExternalLink,
  FileAudio,
  FolderOpen,
  Languages,
  Loader2,
  Pause,
  Play,
  Plus,
  Server,
  Settings,
  Sparkles,
  Terminal,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
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

export function HomePage({ bundle, setBundle }: PageProps) {
  const navigate = useNavigate();
  const [text, setText] = useState(sampleText);
  const [referencePath, setReferencePath] = useState("");
  const [languages, setLanguages] = useState<string[]>(["auto"]);
  const [language, setLanguage] = useState("auto");
  const [audioPath, setAudioPath] = useState("");
  const [step, setStep] = useState<CreateStep>("idle");
  const [status, setStatus] = useState("Ready to generate.");
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
      setStatus("Input text required.");
      return;
    }

    setBusy(true);
    setError("");
    setAudioPath("");
    try {
      setStep("starting");
      setStatus("Initializing Engine...");
      await invoke<RunnerInfo>("start_runner");

      setStep("loading");
      setStatus("Loading models...");
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
      setStatus("Generating audio...");
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
      setStatus("Generation failed.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <AppFrame bundle={bundle}>
      <div className="w-full max-w-[1200px] space-y-10">
        <StudioHeader bundle={bundle} />

        <div className="grid gap-8 lg:grid-cols-[1fr_320px]">
          <div className="space-y-8">
            <EditorCard busy={busy} text={text} setText={setText} createVoice={createVoice} />

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
              setLanguage={setLanguage}
              setReferencePath={setReferencePath}
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
  const [apiUrl, setApiUrl] = useState("");
  const [startingApi, setStartingApi] = useState(false);
  const [copied, setCopied] = useState<"agent" | "curl" | "">("");

  async function openModelsFolder() {
    setError("");
    try {
      const current = bundle ?? (await invoke<ModelBundle>("get_model_bundle"));
      await openPath(current.model_dir);
    } catch (err) {
      setError(String(err));
    }
  }

  async function startApi() {
    setStartingApi(true);
    setError("");
    try {
      const info = await invoke<RunnerInfo>("start_runner");
      setApiUrl(info.base_url);
      return info.base_url;
    } catch (err) {
      setError(String(err));
      return "";
    } finally {
      setStartingApi(false);
    }
  }

  async function openApiDocs() {
    const baseUrl = apiUrl || (await startApi());
    if (baseUrl) await openUrl(`${baseUrl}/docs`);
  }

  async function copyText(kind: "agent" | "curl", text: string) {
    await navigator.clipboard.writeText(text);
    setCopied(kind);
    window.setTimeout(() => setCopied(""), 1600);
  }

  const shownApiUrl = apiUrl || "Start the local API to see the URL";
  const openApiUrl = apiUrl ? `${apiUrl}/openapi.json` : "http://127.0.0.1:<port>/openapi.json";
  const curlExamples = apiUrl
    ? `curl ${apiUrl}/health

curl ${apiUrl}/openapi.json

curl -X POST ${apiUrl}/v1/audio/speech \\
  -H 'Content-Type: application/json' \\
  -o speech.wav \\
  -d '{"input":"Hello from Chirp","language":"auto","response_format":"wav"}'`
    : `curl http://127.0.0.1:<port>/health
curl http://127.0.0.1:<port>/openapi.json`;
  const agentPrompt = `You are using Chirp, a local Qwen3-TTS HTTP API.

Base URL: ${apiUrl || "ask the user for the local Chirp API base URL shown in Settings"}
OpenAPI schema: ${openApiUrl}
Swagger docs: ${apiUrl ? `${apiUrl}/docs` : "open the /docs endpoint from the local base URL"}

Before calling the API, fetch the OpenAPI schema from /openapi.json and use it as the source of truth for request and response shapes.
Use POST /v1/audio/speech to synthesize speech. Send JSON with input, optional voice_reference, language, and response_format set to wav. Save the audio/wav response to a .wav file.
If the API returns no_model, ask the user to load or install the Chirp model in the desktop app first.`;

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

          <div className="space-y-4">
            <h3 className="text-[10px] font-bold uppercase tracking-[0.2em] text-secondary opacity-30">Local API & Agents</h3>
            <Card className="divide-y divide-border/20 overflow-hidden border-none shadow-xl">
              <div className="space-y-4 p-6">
                <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
                  <div className="min-w-0 space-y-1">
                    <p className="flex items-center gap-2 text-base font-semibold tracking-tight text-primary">
                      <Server className="h-4 w-4 text-secondary opacity-40" />
                      Chirp HTTP API
                    </p>
                    <p className="text-xs leading-5 text-secondary opacity-50">Swagger docs, OpenAPI schema, and agent-ready examples.</p>
                  </div>
                  <Button
                    variant={apiUrl ? "secondary" : "primary"}
                    onClick={startApi}
                    disabled={startingApi}
                    className="h-8 shrink-0 gap-2 rounded-full px-3 text-[10px] font-black uppercase tracking-[0.16em]"
                  >
                    {startingApi ? (
                      <Loader2 className="h-3.5 w-3.5 animate-spin" />
                    ) : (
                      <span className={cn("h-1.5 w-1.5 rounded-full", apiUrl ? "bg-green-500" : "bg-white/70")} />
                    )}
                    {apiUrl ? "Running" : "Start API"}
                  </Button>
                </div>

                <p className="truncate rounded-lg border border-border/10 bg-background/50 px-3 py-2 font-mono text-[11px] text-secondary/70">
                  {shownApiUrl}
                </p>

                <div className="grid gap-2 sm:grid-cols-3">
                  <Button variant="outline" onClick={openApiDocs} className="h-9 gap-2 px-3 text-[11px]">
                    <ExternalLink className="h-4 w-4" />
                    Swagger
                  </Button>
                  <Button variant="secondary" onClick={() => copyText("agent", agentPrompt)} className="h-9 gap-2 px-3 text-[11px]">
                    {copied === "agent" ? <Check className="h-4 w-4" /> : <Clipboard className="h-4 w-4" />}
                    {copied === "agent" ? "Copied" : "Agent Skill"}
                  </Button>
                  <Button variant="secondary" onClick={() => copyText("curl", curlExamples)} className="h-9 gap-2 px-3 text-[11px]">
                    {copied === "curl" ? <Check className="h-4 w-4" /> : <Terminal className="h-4 w-4" />}
                    {copied === "curl" ? "Copied" : "cURL"}
                  </Button>
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
    </header>
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
