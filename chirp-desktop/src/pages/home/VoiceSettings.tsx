import { convertFileSrc, invoke } from "@tauri-apps/api/core";
import { AudioLines, ChevronRight, Download, FileAudio, Languages, Loader2, Pause, Play, Plus, UserRound, X } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { cn } from "../../lib/classNames";
import { describeKokoroVoice, describeKokoroVoices, filterKokoroVoices, kokoroVoiceFilters } from "../../lib/kokoroVoices";
import type { KokoroVoiceFilter } from "../../lib/kokoroVoices";
import type { DownloadedVoice, ModelBundle, VoicePreset } from "../../lib/types";
import { voiceCatalog, voiceFilters } from "../../lib/voices";
import type { VoiceFilter } from "../../lib/voices";
import { Button, Card, Eyebrow } from "../../components/ui";

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

export function VoiceSettings({
  busy,
  language,
  languages,
  referencePath,
  runtime,
  kokoroVoice,
  kokoroVoiceIds,
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
  kokoroVoiceIds: string[];
  chooseReference: () => void;
  setLanguage: (language: string) => void;
  setKokoroVoice: (voice: string) => void;
  setReferencePath: (path: string) => void;
}) {
  const referenceSrc = useMemo(() => (referencePath ? convertFileSrc(referencePath) : ""), [referencePath]);
  const [voiceBusy, setVoiceBusy] = useState("");
  const [voiceError, setVoiceError] = useState("");
  const [libraryOpen, setLibraryOpen] = useState(false);
  const kokoroVoiceOptions = useMemo(() => describeKokoroVoices(kokoroVoiceIds.length ? kokoroVoiceIds : [kokoroVoice]), [kokoroVoiceIds, kokoroVoice]);
  const selectedKokoroVoice = kokoroVoiceOptions.find((voice) => voice.id === kokoroVoice) ?? describeKokoroVoice(kokoroVoice);

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
          voices={kokoroVoiceOptions}
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
  voices,
  selectedVoice,
  onClose,
  onChoose,
}: {
  busy: boolean;
  open: boolean;
  voices: ReturnType<typeof describeKokoroVoices>;
  selectedVoice: string;
  onClose: () => void;
  onChoose: (voice: string) => void;
}) {
  const [filter, setFilter] = useState<KokoroVoiceFilter>("all");
  const visibleVoices = useMemo(
    () => filterKokoroVoices(voices, filter),
    [voices, filter],
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
                </div>
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}
