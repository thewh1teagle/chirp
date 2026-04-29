import { Check, Copy, X } from "lucide-react"
import { useState } from "react"

import { Button } from "@/components/ui/button"

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
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-background/80 px-4 backdrop-blur-sm">
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="linux-install-title"
        className="w-full max-w-xl rounded-xl border bg-card p-5 text-left shadow-lg"
      >
        <div className="flex items-center justify-between gap-4">
          <h2 id="linux-install-title" className="text-lg font-semibold">
            Install Chirp on Linux
          </h2>
          <button
            type="button"
            onClick={() => onOpenChange(false)}
            className="inline-flex size-8 items-center justify-center rounded-md text-muted-foreground hover:bg-accent hover:text-foreground"
            aria-label="Close"
          >
            <X className="size-4" />
          </button>
        </div>

        <p className="mt-2 text-sm leading-6 text-muted-foreground">
          Run this command to download the latest desktop AppImage and install it under <code>~/.local/bin</code>.
        </p>

        <div className="mt-4 flex min-w-0 items-center gap-2 rounded-lg bg-[#EFEFEF] p-2 text-[#111111]">
          <code className="min-w-0 flex-1 overflow-x-auto whitespace-nowrap px-2 py-1 text-sm">{command}</code>
          <Button variant="secondary" size="sm" onClick={copyCommand}>
            {copied ? <Check className="size-4" /> : <Copy className="size-4" />}
            {copied ? "Copied" : "Copy"}
          </Button>
        </div>
      </div>
    </div>
  )
}
