import { Check, Copy, X } from "lucide-react"
import { useState } from "react"

import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"

interface LinuxInstallModalProps {
  command: string
  open: boolean
  onOpenChange: (open: boolean) => void
}

export function LinuxInstallModal({ command, open, onOpenChange }: LinuxInstallModalProps) {
  const [copied, setCopied] = useState(false)

  if (!open) return null

  async function copyCommand() {
    await navigator.clipboard.writeText(command)
    setCopied(true)
    window.setTimeout(() => setCopied(false), 1600)
  }

  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/10 px-4 backdrop-blur-md">
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="linux-install-title"
        className="w-full max-w-xl rounded-[28px] border border-border/40 bg-white p-8 text-left shadow-[0_32px_64px_rgb(0,0,0,0.1)] ring-1 ring-black/5"
      >
        <div className="flex items-center justify-between gap-4">
          <h2 id="linux-install-title" className="text-xl font-bold tracking-tight">
            Install on Linux
          </h2>
          <button
            type="button"
            onClick={() => onOpenChange(false)}
            className="inline-flex size-10 items-center justify-center rounded-2xl text-muted-foreground/60 transition-all hover:bg-black/[0.03] hover:text-foreground active:scale-95"
            aria-label="Close"
          >
            <X className="size-5" />
          </button>
        </div>

        <p className="mt-4 text-[15px] leading-relaxed text-muted-foreground">
          Run this command in your terminal to download the latest AppImage and install it locally under <code>~/.local/bin</code>.
        </p>

        <div className="mt-8 flex min-w-0 items-center gap-2 rounded-2xl bg-black/[0.02] p-2 ring-1 ring-black/[0.03]">
          <code className="min-w-0 flex-1 overflow-x-auto whitespace-nowrap px-4 py-2 font-mono text-[13px] text-foreground/80">{command}</code>
          <Button 
            variant="secondary" 
            size="sm" 
            onClick={copyCommand}
            className={cn(
                "h-10 rounded-xl px-4 font-semibold transition-all active:scale-95",
                copied ? "bg-green-500/10 text-green-600 hover:bg-green-500/20" : "bg-white shadow-sm ring-1 ring-black/[0.03] hover:bg-white hover:ring-black/[0.06]"
            )}
          >
            {copied ? <Check className="size-4 mr-1.5" /> : <Copy className="size-4 mr-1.5" />}
            {copied ? "Copied" : "Copy"}
          </Button>
        </div>
      </div>
    </div>
  )
}
