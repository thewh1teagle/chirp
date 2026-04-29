#!/bin/sh
set -eu

repo="thewh1teagle/chirp"
tag="${1:-}"
install_dir="${CHIRP_INSTALL_DIR:-$HOME/.local/bin}"
bin_path="$install_dir/chirp.AppImage"

need() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Missing required command: $1" >&2
    exit 1
  }
}

need curl
need grep
need sed

if [ -z "$tag" ]; then
  tag="$(
    curl -fsSL "https://api.github.com/repos/$repo/releases?per_page=50" |
      grep -o '"tag_name": *"chirp-desktop-v[^"]*"' |
      sed -E 's/.*"([^"]+)"/\1/' |
      head -n 1
  )"
fi

if [ -z "$tag" ]; then
  echo "Could not find latest chirp desktop release." >&2
  exit 1
fi

asset="$(
  curl -fsSL "https://api.github.com/repos/$repo/releases/tags/$tag" |
    grep -o '"browser_download_url": *"[^"]*\.AppImage"' |
    sed -E 's/.*"([^"]+)"/\1/' |
    head -n 1
)"

if [ -z "$asset" ]; then
  echo "Could not find Linux AppImage asset for $tag." >&2
  exit 1
fi

mkdir -p "$install_dir"
echo "Downloading Chirp $tag..."
curl -fL "$asset" -o "$bin_path"
chmod +x "$bin_path"

echo "Installed Chirp to $bin_path"
echo "Run it with: $bin_path"
