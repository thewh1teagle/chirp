#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import platform
import tarfile
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

HOST_TRIPLE_MAP = {
    ("Darwin", "arm64"): "aarch64-apple-darwin",
    ("Darwin", "x86_64"): "x86_64-apple-darwin",
    ("Linux", "x86_64"): "x86_64-unknown-linux-gnu",
    ("Linux", "aarch64"): "aarch64-unknown-linux-gnu",
    ("Windows", "AMD64"): "x86_64-pc-windows-msvc",
}

RUNNER_ASSET_MAP = {
    "aarch64-apple-darwin": ("chirp-runner-darwin-arm64.tar.gz", "chirp"),
    "x86_64-unknown-linux-gnu": ("chirp-runner-linux-x64.tar.gz", "chirp"),
    "x86_64-pc-windows-msvc": ("chirp-runner-windows-x64.zip", "chirp.exe"),
}


def detect_host_target() -> str | None:
    return HOST_TRIPLE_MAP.get((platform.system(), platform.machine()))


def download(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read()


def extract_member(data: bytes, archive_name: str, member_name: str) -> bytes:
    if archive_name.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for name in zf.namelist():
                if Path(name).name == member_name:
                    return zf.read(name)
    else:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
            for member in tf.getmembers():
                if Path(member.name).name == member_name:
                    extracted = tf.extractfile(member)
                    if extracted is not None:
                        return extracted.read()
    raise RuntimeError(f"{member_name} not found in {archive_name}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download the Chirp runner sidecar for Tauri builds")
    parser.add_argument("--target", help="Rust target triple, for example x86_64-unknown-linux-gnu")
    parser.add_argument("--repo", default="thewh1teagle/chirp", help="GitHub repo owner/name")
    parser.add_argument("--version-file", type=Path, default=ROOT / ".runner-version")
    args = parser.parse_args()

    target = args.target or detect_host_target()
    if not target:
        print("Warning: could not detect host target; pass --target or place the runner sidecar manually.")
        return 0

    asset = RUNNER_ASSET_MAP.get(target)
    if not asset:
        print(f"Warning: no Chirp runner release asset is configured for target {target}.")
        return 0

    tag = args.version_file.read_text(encoding="utf-8").strip()
    if not tag:
        print(f"Warning: {args.version_file} is empty; skipping runner sidecar download.")
        return 0

    archive_name, member_name = asset
    is_windows = target.endswith("windows-msvc")
    sidecar_name = f"chirp-runner-{target}" + (".exe" if is_windows else "")
    dest_dir = ROOT / "desktop" / "src-tauri" / "binaries"
    dest = dest_dir / sidecar_name

    if dest.exists():
        print(f"Chirp runner sidecar already exists at {dest}; skipping download.")
        return 0

    url = f"https://github.com/{args.repo}/releases/download/{tag}/{archive_name}"
    print(f"Downloading {url}")
    data = download(url)
    binary = extract_member(data, archive_name, member_name)

    dest_dir.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(binary)
    if not is_windows:
        dest.chmod(dest.stat().st_mode | 0o111)
    print(f"Installed Chirp runner sidecar at {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
