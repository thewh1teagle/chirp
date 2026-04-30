import { Loader2, Play } from "lucide-react";
import { cn } from "../../lib/classNames";
import { Button, Card } from "../../components/ui";

type EditorCardProps = {
  busy: boolean;
  text: string;
  setText: (text: string) => void;
  createVoice: () => void;
};

export function EditorCard({ busy, text, setText, createVoice }: EditorCardProps) {
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
