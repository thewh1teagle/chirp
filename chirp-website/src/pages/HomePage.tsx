import chirpLogo from "@/assets/chirp-logo.svg"
import { DownloadCta } from "@/components/DownloadCta"
import { SiteFooter } from "@/components/SiteFooter"
import { SiteHeader } from "@/components/SiteHeader"

export function HomePage() {
  return (
    <div className="flex min-h-svh flex-col bg-[radial-gradient(circle_at_top,hsl(196_85%_96%),transparent_34rem)]">
      <SiteHeader />

      <main className="mx-auto flex w-full max-w-5xl flex-1 flex-col items-center justify-center px-4 py-16 text-center sm:px-6">
        <img src={chirpLogo} alt="" className="mb-8 size-20 rounded-2xl shadow-sm" />
        <h1 className="max-w-3xl text-4xl font-semibold tracking-tight text-balance sm:text-6xl">
          Offline text to speech for your desktop.
        </h1>
        <p className="mt-5 max-w-2xl text-base leading-8 text-muted-foreground sm:text-lg">
          Chirp runs locally, generates natural speech, and keeps your audio workflow on your machine.
        </p>

        <DownloadCta />
      </main>

      <SiteFooter />
    </div>
  )
}
