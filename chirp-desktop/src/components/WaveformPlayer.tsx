import { Pause, Play } from "lucide-react";
import { useMemo, useRef, useState } from "react";
import { Card } from "./ui";

export function WaveformPlayer({ src, filename }: { src: string; filename: string }) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [progress, setProgress] = useState(0);
  const [duration, setDuration] = useState(0);

  const togglePlay = () => {
    if (!audioRef.current) return;
    if (isPlaying) audioRef.current.pause();
    else audioRef.current.play();
    setIsPlaying(!isPlaying);
  };

  const onTimeUpdate = () => {
    if (audioRef.current) {
      setProgress(audioRef.current.currentTime / audioRef.current.duration);
    }
  };

  const onLoadedMetadata = () => {
    if (audioRef.current) setDuration(audioRef.current.duration);
  };

  const seek = (event: React.MouseEvent<HTMLDivElement>) => {
    if (audioRef.current && audioRef.current.duration) {
      const rect = event.currentTarget.getBoundingClientRect();
      const pct = (event.clientX - rect.left) / rect.width;
      audioRef.current.currentTime = pct * audioRef.current.duration;
    }
  };

  const formatTime = (seconds: number) => {
    if (isNaN(seconds)) return "0:00";
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  const bars = useMemo(() => Array.from({ length: 50 }, () => 20 + Math.random() * 70), []);

  return (
    <Card className="flex flex-col gap-4 border-none bg-surface p-6 shadow-2xl sm:flex-row sm:items-center sm:gap-8">
      <audio
        ref={audioRef}
        src={src}
        onTimeUpdate={onTimeUpdate}
        onLoadedMetadata={onLoadedMetadata}
        onEnded={() => setIsPlaying(false)}
        autoPlay
      />

      <div className="flex items-center gap-4">
        <button
          onClick={togglePlay}
          className="flex h-14 w-14 shrink-0 items-center justify-center rounded-full bg-primary text-white transition-all hover:scale-105 active:scale-95 shadow-lg shadow-primary/20"
        >
          {isPlaying ? <Pause className="h-6 w-6 fill-current" /> : <Play className="h-6 w-6 fill-current ml-1" />}
        </button>
        <div className="min-w-0">
          <p className="text-sm font-bold text-primary">Preview Ready</p>
          <p className="truncate text-xs font-medium text-secondary opacity-60">{filename}</p>
        </div>
      </div>

      <div className="flex flex-1 flex-col gap-2">
        <div className="relative flex h-14 cursor-pointer items-end gap-[2px]" onClick={seek}>
          {bars.map((height, index) => (
            <div
              key={index}
              className="flex-1 rounded-full transition-all duration-300"
              style={{
                height: `${height}%`,
                backgroundColor: index / bars.length < progress ? "var(--color-primary)" : "var(--color-border)",
              }}
            />
          ))}
        </div>
        <div className="flex justify-between text-[10px] font-bold uppercase tracking-widest text-secondary opacity-40">
          <span>{formatTime(audioRef.current?.currentTime || 0)}</span>
          <span>{formatTime(duration)}</span>
        </div>
      </div>
    </Card>
  );
}
