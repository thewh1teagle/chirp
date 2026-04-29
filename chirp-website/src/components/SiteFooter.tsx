import { githubUrl, issuesUrl, releasesUrl } from "@/lib/links"

export function SiteFooter() {
  return (
    <footer className="mx-auto w-full max-w-5xl px-4 py-6 text-sm text-muted-foreground sm:px-6">
      <div className="flex flex-col items-center justify-between gap-4 border-t pt-6 sm:flex-row">
        <span>Chirp</span>
        <nav className="flex items-center gap-5">
          <a className="transition-colors hover:text-foreground" href={githubUrl} target="_blank" rel="noreferrer">
            GitHub
          </a>
          <a className="transition-colors hover:text-foreground" href={releasesUrl}>
            Releases
          </a>
          <a className="transition-colors hover:text-foreground" href={issuesUrl}>
            Issues
          </a>
        </nav>
      </div>
    </footer>
  )
}
