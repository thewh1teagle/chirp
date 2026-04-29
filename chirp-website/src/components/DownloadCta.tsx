import { Monitor, Smartphone } from "lucide-react"
import { useMemo, useState } from "react"

import { GithubIcon } from "@/components/icons/GithubIcon"
import { PlatformIcon } from "@/components/icons/PlatformIcon"
import { LinuxInstallModal } from "@/components/LinuxInstallModal"
import { Button } from "@/components/ui/button"
import { githubUrl, releasesUrl } from "@/lib/links"
import latestReleaseJson from "@/lib/latest_release.json"
import type { LatestRelease, ReleaseAsset } from "@/lib/latestRelease"
import { detectPlatform, isMobileDevice, type Platform, platformLabels } from "@/lib/platform"
import { cn } from "@/lib/utils"

const latestRelease = latestReleaseJson as LatestRelease
const installCommand = `curl -sSf https://thewh1teagle.github.io/chirp/installer.sh | sh -s ${latestRelease.version}`

function preferredAsset(platform: Platform): ReleaseAsset | undefined {
  const assets = latestRelease.assets.filter((asset) => asset.platform === platform)

  if (platform === "windows") {
    return assets.find((asset) => asset.kind === "exe") ?? assets.find((asset) => asset.kind === "msi")
  }

  if (platform === "linux") {
    return assets.find((asset) => asset.kind === "appimage") ?? assets.find((asset) => asset.kind === "deb") ?? assets[0]
  }

  return assets.find((asset) => asset.arch === "darwin-aarch64") ?? assets[0]
}

export function DownloadCta() {
  const [platform, setPlatform] = useState<Platform>(() => detectPlatform())
  const [isMobile, setIsMobile] = useState(() => isMobileDevice())
  const [linuxModalOpen, setLinuxModalOpen] = useState(false)
  const asset = preferredAsset(platform)

  const downloadLabel = useMemo(() => {
    if (isMobile) return "Download on desktop"

    return `Download for ${platformLabels[platform]}`
  }, [isMobile, platform])

  return (
    <>
      <div className="mt-9 flex w-full max-w-md flex-col items-stretch justify-center gap-3 sm:max-w-none sm:flex-row">
        {platform === "linux" && !isMobile ? (
          <Button size="lg" onClick={() => setLinuxModalOpen(true)}>
            <PlatformIcon platform={platform} className="size-4" />
            {downloadLabel}
          </Button>
        ) : (
          <Button size="lg" asChild>
            <a href={isMobile ? releasesUrl : asset?.url ?? releasesUrl}>
              {isMobile ? <Smartphone className="size-4" /> : <PlatformIcon platform={platform} className="size-4" />}
              {downloadLabel}
            </a>
          </Button>
        )}
        <Button variant="outline" size="lg" asChild>
          <a href={githubUrl} target="_blank" rel="noreferrer">
            <GithubIcon className="size-4" />
            View on GitHub
          </a>
        </Button>
      </div>

      <div className="mt-5 flex items-center justify-center gap-1 rounded-lg border bg-card/70 p-1">
        {(Object.keys(platformLabels) as Platform[]).map((option) => (
          <button
            key={option}
            type="button"
            onClick={() => {
              setPlatform(option)
              setIsMobile(false)
            }}
            className={cn(
              "inline-flex size-9 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-accent hover:text-foreground",
              platform === option && !isMobile && "bg-accent text-foreground"
            )}
            aria-label={`Select ${platformLabels[option]}`}
          >
            <PlatformIcon platform={option} />
          </button>
        ))}
      </div>

      <div className="mt-4 flex items-center gap-2 text-sm text-muted-foreground">
        {isMobile ? <Smartphone className="size-4" /> : <Monitor className="size-4" />}
        <span>
          {isMobile
            ? "Available for macOS, Windows, and Linux."
            : `${latestRelease.version.replace("chirp-desktop-", "")} for ${platformLabels[platform]}.`}
        </span>
      </div>

      <LinuxInstallModal command={installCommand} open={linuxModalOpen} onOpenChange={setLinuxModalOpen} />
    </>
  )
}
