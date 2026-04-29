import { invoke } from "@tauri-apps/api/core";
import { save } from "@tauri-apps/plugin-dialog";
import { Download, Pause, Play } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Button, Card } from "./ui";
import { motion } from "framer-motion";

export function WaveformPlayer({
  src,
  sourcePath,
  filename,
  autoPlayOnce = false,
  onAutoPlayConsumed,
}: {
  src: string;
  sourcePath: string;
  filename: string;
  autoPlayOnce?: boolean;
  onAutoPlayConsumed?: () => void;
}) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const waveformRef = useRef<HTMLDivElement>(null);
  const draggingRef = useRef(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [progress, setProgress] = useState(0);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [isDragging, setIsDragging] = useState(false);
  const [downloading, setDownloading] = useState(false);
  const [savedPath, setSavedPath] = useState("");
  const [downloadError, setDownloadError] = useState("");

  useEffect(() => {
    if (!autoPlayOnce || !audioRef.current) return;
    audioRef.current.play().catch(() => setIsPlaying(false));
    onAutoPlayConsumed?.();
  }, [autoPlayOnce, onAutoPlayConsumed, src]);

  const togglePlay = () => {
    if (!audioRef.current) return;
    if (isPlaying) audioRef.current.pause();
    else {
      audioRef.current.play().catch(() => setIsPlaying(false));
    }
  };

  const downloadAudio = async () => {
    setDownloading(true);
    setDownloadError("");
    try {
      const destinationPath = await save({
        defaultPath: filename,
        filters: [{ name: "WAV audio", extensions: ["wav"] }],
      });
      if (!destinationPath) return;

      await invoke("copy_audio_file", {
        sourcePath,
        destinationPath,
      });
      setSavedPath(destinationPath);
    } catch (err) {
      setDownloadError(String(err));
    } finally {
      setDownloading(false);
    }
  };

  const revealSavedAudio = async () => {
    if (!savedPath) return;
    setDownloadError("");
    try {
      await invoke("reveal_path", { path: savedPath });
    } catch (err) {
      setDownloadError(String(err));
    }
  };

  const onTimeUpdate = () => {
    if (audioRef.current && !draggingRef.current) {
      setCurrentTime(audioRef.current.currentTime);
      setProgress(audioRef.current.currentTime / audioRef.current.duration);
    }
  };

  const onLoadedMetadata = () => {
    if (audioRef.current) setDuration(audioRef.current.duration);
  };

  const seekToClientX = useCallback((clientX: number) => {
    if (!waveformRef.current || !audioRef.current || !duration) return;
    const rect = waveformRef.current.getBoundingClientRect();
    const x = Math.max(0, Math.min(clientX - rect.left, rect.width));
    const nextProgress = x / rect.width;
    const nextTime = nextProgress * duration;

    audioRef.current.currentTime = nextTime;
    setCurrentTime(nextTime);
    setProgress(nextProgress);
  }, [duration]);

  const handlePointerDown = (e: React.PointerEvent) => {
    e.preventDefault();
    draggingRef.current = true;
    setIsDragging(true);
    seekToClientX(e.clientX);
    waveformRef.current?.setPointerCapture(e.pointerId);
  };

  const handlePointerMove = (e: React.PointerEvent) => {
    if (!draggingRef.current) return;
    e.preventDefault();
    seekToClientX(e.clientX);
  };

  const handlePointerUp = (e: React.PointerEvent) => {
    if (!draggingRef.current) return;
    seekToClientX(e.clientX);
    draggingRef.current = false;
    setIsDragging(false);
    if (waveformRef.current?.hasPointerCapture(e.pointerId)) {
      waveformRef.current.releasePointerCapture(e.pointerId);
    }
  };

  const formatTime = (seconds: number) => {
    if (isNaN(seconds)) return "0:00";
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  const bars = useMemo(() => Array.from({ length: 50 }, () => 20 + Math.random() * 60), []);

  return (
    <Card className="group flex flex-col gap-5 border-none bg-white p-5 shadow-2xl transition-all hover:shadow-3xl sm:flex-row sm:items-center sm:gap-8">
      <audio
        ref={audioRef}
        src={src}
        onTimeUpdate={onTimeUpdate}
        onLoadedMetadata={onLoadedMetadata}
        onPlay={() => setIsPlaying(true)}
        onPause={() => setIsPlaying(false)}
        onEnded={() => setIsPlaying(false)}
      />

      <div className="flex items-center gap-4">
        <button
          onClick={togglePlay}
          className="flex h-12 w-12 shrink-0 items-center justify-center rounded-full bg-primary text-white transition-all hover:scale-105 active:scale-95 shadow-lg shadow-primary/10 cursor-pointer"
        >
          {isPlaying ? <Pause className="h-5 w-5 fill-current" /> : <Play className="h-5 w-5 fill-current ml-0.5" />}
        </button>
        <div className="min-w-0">
          <p className="text-sm font-bold tracking-tight text-primary">Preview</p>
          <p className="truncate text-[10px] font-bold uppercase tracking-widest text-secondary opacity-30">{filename}</p>
        </div>
      </div>

      <div className="flex flex-1 flex-col gap-2">
        <div 
          ref={waveformRef}
          className="relative flex h-10 cursor-pointer items-center gap-[2px] touch-none" 
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
          onPointerCancel={handlePointerUp}
        >
          {bars.map((height, index) => (
            <motion.div
              key={index}
              initial={false}
              animate={{
                backgroundColor: index / bars.length < progress ? "var(--color-primary)" : "var(--color-border)",
                opacity: index / bars.length < progress ? 1 : 0.8,
              }}
              transition={{ duration: isDragging ? 0 : 0.12 }}
              className="flex-1 rounded-full"
              style={{
                height: `${height}%`,
              }}
            />
          ))}
        </div>
        <div className="flex justify-between font-mono text-[9px] font-bold uppercase tracking-widest text-secondary opacity-30">
          <span>{formatTime(currentTime)}</span>
          <span>{formatTime(duration)}</span>
        </div>
      </div>

      <div className="flex shrink-0 items-center gap-2 border-l border-border/10 pl-6">
        <Button 
          variant="outline" 
          onClick={downloadAudio} 
          disabled={downloading}
          className="h-9 w-9 p-0 rounded-full transition-transform hover:scale-110 active:scale-90"
          title="Save audio"
        >
          <Download className="h-4 w-4" />
        </Button>
      </div>

      {(savedPath || downloadError) && (
        <div className="basis-full rounded-xl border border-border/40 bg-background/40 p-3 text-xs">
          {downloadError ? (
            <p className="font-medium text-red-800">{downloadError}</p>
          ) : (
            <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <p className="min-w-0 truncate font-medium text-secondary">Saved to {savedPath}</p>
              <Button variant="outline" onClick={revealSavedAudio} className="h-8 shrink-0 px-3 text-xs">
                Show in Finder
              </Button>
            </div>
          )}
        </div>
      )}
    </Card>
  );
}
