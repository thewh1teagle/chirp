import chirpLogo from "@/assets/chirp-logo.svg"
import { GithubIcon } from "@/components/icons/GithubIcon"
import { githubUrl } from "@/lib/links"

export function SiteHeader() {
  return (
    <header className="mx-auto flex w-full max-w-5xl items-center justify-between px-4 py-4 sm:px-6">
      <a href="/" className="inline-flex items-center gap-2 text-base font-medium">
        <img src={chirpLogo} alt="" className="size-8 rounded-lg" />
        <span>Chirp</span>
      </a>
      <a
        href={githubUrl}
        target="_blank"
        rel="noreferrer"
        className="inline-flex size-9 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
        aria-label="Chirp on GitHub"
      >
        <GithubIcon className="size-5" />
      </a>
    </header>
  )
}
