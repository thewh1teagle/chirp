import { convertFileSrc, invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { open } from "@tauri-apps/plugin-dialog";
import { openPath } from "@tauri-apps/plugin-opener";
import { ComponentProps, ReactNode, useEffect, useMemo, useState } from "react";
import { Link, Navigate, useLocation, useNavigate } from "react-router";
import "./App.css";

type ModelBundle = {
  installed: boolean;
  model_path: string;
  codec_path: string;
  model_dir: string;
  version: string;
  url: string;
};

type RunnerInfo = {
  base_url: string;
};

type DownloadProgress = {
  downloaded: number;
  total?: number | null;
  progress?: number | null;
  stage: "downloading" | "extracting";
};

type CreateStep = "idle" | "starting" | "loading" | "creating" | "done";
type ButtonVariant = "primary" | "secondary" | "ghost";

const sampleText =
  "Create a warm, natural voice preview from this text. Chirp runs locally and keeps the model on this device.";

const formatBytes = (bytes: number) => {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 MB";
  return `${(bytes / 1024 / 1024).toFixed(bytes > 1024 * 1024 * 1024 ? 2 : 0)} MB`;
};

const join = (...classes: Array<string | false | null | undefined>) => classes.filter(Boolean).join(" ");

function App() {
  const location = useLocation();
  const navigate = useNavigate();
  const [bundle, setBundle] = useState<ModelBundle | null>(null);
  const [checking, setChecking] = useState(true);

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
      <main className="grid min-h-screen place-items-center bg-[#f7f7f4] p-8 text-[#171717]">
        <div className="space-y-4 text-center">
          <Brand />
          <p className="text-sm text-[#68645c]">Preparing your local voice studio...</p>
        </div>
      </main>
    );
  }

  if (location.pathname === "/onboard") return <OnboardPage bundle={bundle} setBundle={setBundle} />;
  if (location.pathname === "/home") return <HomePage bundle={bundle} setBundle={setBundle} />;
  if (location.pathname === "/settings") return <SettingsPage bundle={bundle} />;

  return <Navigate to={bundle?.installed ? "/home" : "/onboard"} replace />;
}

function OnboardPage({
  bundle,
  setBundle,
}: {
  bundle: ModelBundle | null;
  setBundle: (bundle: ModelBundle) => void;
}) {
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
    <main className="grid min-h-screen place-items-center bg-[#f7f7f4] p-5 text-[#171717] sm:p-8">
      <section className="w-full max-w-[680px]">
        <Brand />
        <h1 className="mt-7 max-w-[560px] text-[34px] font-semibold leading-[1.04] tracking-normal text-[#11110f] sm:text-5xl">
          Set up local voice creation
        </h1>
        <p className="mt-3 max-w-[590px] text-base leading-7 text-[#68645c] sm:text-lg">
          Chirp downloads the Qwen3-TTS voice model once, stores it on this device, and uses it locally after that.
        </p>

        <Card className="mt-7 flex flex-col gap-5 p-5 sm:flex-row sm:items-center sm:justify-between">
          <div className="min-w-0">
            <Eyebrow>One-time download</Eyebrow>
            <strong className="block break-words text-[#171717]">{bundle?.version ?? "chirp-models-v0.1.3"}</strong>
            <p className="mt-2 break-words text-sm text-[#68645c]">{bundle?.model_dir ?? "Stored in your app data folder"}</p>
          </div>
          <Button onClick={downloadModel} disabled={busy}>
            {busy ? "Downloading..." : "Download model"}
          </Button>
        </Card>

        {(busy || progress) && (
          <Card className="mt-4 p-4 shadow-none">
            <div className="mb-3 flex items-center justify-between gap-4 text-sm font-semibold text-[#2b2a26]">
              <span>{stageLabel}</span>
              <strong>{progress?.stage === "extracting" ? "Almost done" : `${progressValue}%`}</strong>
            </div>
            <Progress value={progress?.stage === "extracting" ? 100 : progressValue} />
            <p className="mt-3 text-sm text-[#68645c]">
              {progress?.total ? `${formatBytes(progress.downloaded)} of ${formatBytes(progress.total)}` : "Preparing download..."}
            </p>
          </Card>
        )}

        {error && <ErrorBlock>{error}</ErrorBlock>}
      </section>
    </main>
  );
}

function HomePage({
  bundle,
  setBundle,
}: {
  bundle: ModelBundle | null;
  setBundle: (bundle: ModelBundle) => void;
}) {
  const navigate = useNavigate();
  const [runnerUrl, setRunnerUrl] = useState("");
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
      setStatus("Starting local voice engine...");
      const info = await invoke<RunnerInfo>("start_runner");
      setRunnerUrl(info.base_url);

      setStep("loading");
      setStatus("Loading voice model...");
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
      setStatus("Creating audio...");
      const output = await invoke<string>("synthesize", {
        request: {
          input: text,
          voice_reference: referencePath || null,
          language: selectedLanguage,
        },
      });
      setAudioPath(output);
      setStep("done");
      setStatus("Audio is ready.");
    } catch (err) {
      setStep("idle");
      setError(String(err));
      setStatus("Could not create audio.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <AppFrame bundle={bundle}>
      <section className="w-full max-w-[980px]">
        <header className="mb-6 flex flex-col gap-5 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <Brand className="mb-4 h-8 min-w-14 text-[13px]" />
            <h1 className="text-[34px] font-semibold leading-[1.05] tracking-normal text-[#11110f] sm:text-[42px]">Create natural speech</h1>
            <p className="mt-3 text-base text-[#68645c]">Local Qwen3-TTS voice creation with optional reference WAV cloning.</p>
          </div>
          <Card className="min-w-0 p-4 sm:min-w-[210px]">
            <Eyebrow>Model</Eyebrow>
            <strong className="block break-words">{bundle?.version ?? "Checking..."}</strong>
          </Card>
        </header>

        <Card className="p-5">
          <FieldLabel htmlFor="text">Script</FieldLabel>
          <textarea
            id="text"
            value={text}
            onChange={(event) => setText(event.currentTarget.value)}
            disabled={busy}
            className="min-h-[230px] w-full resize-y rounded-lg border border-[#d4d2ca] bg-[#fbfbf8] p-4 text-[17px] leading-7 text-[#171717] outline-none focus:ring-2 focus:ring-[#171717]"
          />

          <div className="mt-3 grid gap-3 md:grid-cols-[minmax(0,1.4fr)_minmax(220px,0.6fr)]">
            <div className="min-w-0 rounded-lg border border-[#e2dfd5] bg-[#fbfbf8] p-4">
              <Eyebrow>Voice reference</Eyebrow>
              <strong className="block break-words">{referencePath ? referencePath.split(/[\\/]/).pop() : "Default voice"}</strong>
              <div className="mt-3 flex flex-wrap gap-2">
                <Button variant="secondary" onClick={chooseReference} disabled={busy}>
                  Choose WAV
                </Button>
                <Button variant="ghost" onClick={() => setReferencePath("")} disabled={busy || !referencePath}>
                  Clear
                </Button>
              </div>
            </div>

            <div className="rounded-lg border border-[#e2dfd5] bg-[#fbfbf8] p-4">
              <FieldLabel htmlFor="language">Language</FieldLabel>
              <select
                id="language"
                value={language}
                onChange={(event) => setLanguage(event.currentTarget.value)}
                disabled={busy}
                className="h-11 w-full rounded-md border border-[#d4d2ca] bg-white px-3 text-[#171717] outline-none focus:ring-2 focus:ring-[#171717]"
              >
                {languages.map((item) => (
                  <option key={item} value={item}>
                    {item === "auto" ? "Auto detect" : item[0].toUpperCase() + item.slice(1)}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {busy && <CreateProgress step={step} />}

          <div className="mt-4 flex flex-col gap-3 sm:flex-row sm:items-center">
            <Button className="min-w-[150px]" onClick={createVoice} disabled={busy || !text.trim()}>
              {busy ? "Creating..." : "Create audio"}
            </Button>
            <span className="text-sm text-[#68645c]">{status}</span>
          </div>
        </Card>

        {audioSrc && (
          <Card className="mt-4 grid gap-4 p-4 md:grid-cols-[180px_minmax(0,1fr)] md:items-center">
            <div>
              <Eyebrow>Preview</Eyebrow>
              <strong>Your audio is ready</strong>
              <p className="mt-2 break-words text-sm text-[#68645c]">{audioPath}</p>
            </div>
            <audio className="w-full" src={audioSrc} controls />
          </Card>
        )}

        {error && <ErrorBlock>{error}</ErrorBlock>}
        {runnerUrl && <p className="mt-4 text-right text-sm text-[#68645c]">Local engine running at {runnerUrl}</p>}
      </section>
    </AppFrame>
  );
}

function SettingsPage({ bundle }: { bundle: ModelBundle | null }) {
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
      <section className="w-full max-w-[760px]">
        <header className="mb-6">
          <Brand className="mb-4 h-8 min-w-14 text-[13px]" />
          <h1 className="text-[34px] font-semibold leading-tight text-[#11110f]">Settings</h1>
          <p className="mt-2 text-base text-[#68645c]">Manage local model files and app storage.</p>
        </header>

        <Card className="p-5">
          <div className="flex flex-col gap-5 sm:flex-row sm:items-center sm:justify-between">
            <div className="min-w-0">
              <Eyebrow>Models folder</Eyebrow>
              <strong className="block break-words text-[#171717]">{bundle?.model_dir ?? "Resolving app data folder..."}</strong>
              <p className="mt-2 text-sm text-[#68645c]">Open the folder where Chirp stores the downloaded GGUF model and codec.</p>
            </div>
            <Button variant="secondary" onClick={openModelsFolder}>
              Open folder
            </Button>
          </div>
        </Card>

        {error && <ErrorBlock>{error}</ErrorBlock>}
      </section>
    </AppFrame>
  );
}

function AppFrame({ bundle, children }: { bundle: ModelBundle | null; children: ReactNode }) {
  return (
    <main className="min-h-screen bg-[#f7f7f4] p-5 text-[#171717] sm:p-8">
      <nav className="mx-auto mb-6 flex w-full max-w-[980px] items-center justify-end gap-2">
        <NavLink to="/home">Home</NavLink>
        <NavLink to="/settings">Settings</NavLink>
        {!bundle?.installed && <NavLink to="/onboard">Setup</NavLink>}
      </nav>
      <div className="mx-auto flex w-full justify-center">{children}</div>
    </main>
  );
}

function CreateProgress({ step }: { step: CreateStep }) {
  const steps: Array<[CreateStep, string]> = [
    ["starting", "Start engine"],
    ["loading", "Load model"],
    ["creating", "Create audio"],
  ];
  const activeIndex = Math.max(
    0,
    steps.findIndex(([key]) => key === step),
  );

  return (
    <div className="mt-4 rounded-lg border border-[#dddacf] bg-[#fbfbf8] p-4">
      <div className="mb-3 flex items-center justify-between gap-3 text-sm">
        <strong className="text-[#171717]">{steps[activeIndex]?.[1] ?? "Create audio"}</strong>
        <span className="text-[#68645c]">Working...</span>
      </div>
      <div className="h-2.5 overflow-hidden rounded-full bg-[#ebe8df]">
        <div className="h-full w-1/3 animate-[loading-slide_1.15s_ease-in-out_infinite] rounded-full bg-[#171717]" />
      </div>
      <div className="mt-3 grid gap-2 md:grid-cols-3">
        {steps.map(([key, label], index) => (
          <div
            key={key}
            className={join(
              "flex items-center gap-2 rounded-md border bg-white p-2.5 text-xs",
              index <= activeIndex ? "border-[#171717] text-[#171717]" : "border-[#e2dfd5] text-[#817c70]",
            )}
          >
            <span className={join("h-2 w-2 rounded-full", index <= activeIndex ? "bg-[#171717]" : "bg-[#c9c5b9]")} />
            <strong>{label}</strong>
          </div>
        ))}
      </div>
    </div>
  );
}

function Button({ className, variant = "primary", ...props }: ComponentProps<"button"> & { variant?: ButtonVariant }) {
  return (
    <button
      className={join(
        "inline-flex h-11 items-center justify-center rounded-md px-4 text-sm font-semibold transition disabled:cursor-not-allowed disabled:opacity-50",
        variant === "primary" && "border border-[#171717] bg-[#171717] text-white hover:bg-black",
        variant === "secondary" && "border border-[#d4d2ca] bg-white text-[#171717] hover:bg-[#fbfbf8]",
        variant === "ghost" && "border border-transparent bg-transparent text-[#68645c] hover:bg-[#eeeae1]",
        className,
      )}
      {...props}
    />
  );
}

function Card({ className, ...props }: ComponentProps<"section">) {
  return <section className={join("rounded-lg border border-[#dddacf] bg-white shadow-[0_18px_50px_rgba(28,28,23,0.06)]", className)} {...props} />;
}

function Progress({ value }: { value: number }) {
  return (
    <div className="h-2.5 w-full overflow-hidden rounded-full bg-[#ebe8df]">
      <div className="h-full rounded-full bg-[#171717] transition-all" style={{ width: `${Math.max(0, Math.min(100, value))}%` }} />
    </div>
  );
}

function Brand({ className }: { className?: string }) {
  return (
    <div
      className={join(
        "inline-grid h-9 min-w-[70px] place-items-center rounded-md border border-[#22221f] px-3 text-sm font-extrabold tracking-normal text-[#171717]",
        className,
      )}
    >
      Chirp
    </div>
  );
}

function Eyebrow({ children }: { children: ReactNode }) {
  return <span className="mb-2 block text-xs font-bold uppercase tracking-[0.04em] text-[#757166]">{children}</span>;
}

function FieldLabel({ className, ...props }: ComponentProps<"label">) {
  return <label className={join("mb-2 block text-xs font-bold uppercase tracking-[0.04em] text-[#757166]", className)} {...props} />;
}

function ErrorBlock({ children }: { children: ReactNode }) {
  return <pre className="mt-4 overflow-auto whitespace-pre-wrap break-words rounded-lg border border-[#e1c7c2] bg-white p-4 text-sm text-[#8a1f11]">{children}</pre>;
}

function NavLink({ to, children }: { to: string; children: ReactNode }) {
  return (
    <Link className="rounded-md border border-[#dddacf] bg-white px-3 py-2 text-sm font-semibold text-[#171717] hover:bg-[#fbfbf8]" to={to}>
      {children}
    </Link>
  );
}

export default App;
