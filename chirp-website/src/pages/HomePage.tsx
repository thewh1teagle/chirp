import chirpLogo from "@/assets/chirp-logo.svg"
import { DownloadCta } from "@/components/DownloadCta"
import { SiteFooter } from "@/components/SiteFooter"
import { SiteHeader } from "@/components/SiteHeader"

export function HomePage() {
  return (
    <div className="flex min-h-svh flex-col bg-[#FBFBFA]">
      <SiteHeader />

      <main className="flex-1 flex flex-col items-center justify-center relative overflow-hidden">
        {/* Decorative Background Element */}
        <div className="absolute top-12 left-1/2 -translate-x-1/2 w-[900px] h-[300px] bg-sky-100/50 rotate-[-10deg] blur-[100px] pointer-events-none" />

        <section className="mx-auto flex w-full max-w-5xl flex-col items-center px-4 py-24 text-center sm:px-6 relative z-10">
          <img src={chirpLogo} alt="" className="mb-8 size-20 rounded-2xl shadow-xl border border-neutral-200" />
          
          <h1 className="max-w-3xl text-5xl font-semibold tracking-tight text-balance sm:text-6xl text-[#111111]">
            Offline text to speech for your desktop.
          </h1>
          
          <p className="mt-6 max-w-2xl text-lg leading-8 text-[#68645C] sm:text-lg">
            Chirp runs locally, generates natural speech, and keeps your audio workflow on your machine.
          </p>

          <div className="mt-12">
            <DownloadCta />
          </div>
        </section>
      </main>

      <SiteFooter />
    </div>
  )
}
