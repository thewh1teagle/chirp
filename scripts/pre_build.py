#!/usr/bin/env python3
from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

HOST_TRIPLE_MAP = {
    ("Darwin", "arm64"): "aarch64-apple-darwin",
    ("Darwin", "x86_64"): "x86_64-apple-darwin",
    ("Linux", "x86_64"): "x86_64-unknown-linux-gnu",
    ("Linux", "aarch64"): "aarch64-unknown-linux-gnu",
    ("Windows", "AMD64"): "x86_64-pc-windows-msvc",
}

def detect_host_target() -> str | None:
    return HOST_TRIPLE_MAP.get((platform.system(), platform.machine()))


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the Chirp server sidecar for Tauri builds")
    parser.add_argument("--target", help="Rust target triple, for example x86_64-unknown-linux-gnu")
    parser.add_argument("--profile", default="release", choices=["debug", "release"])
    args = parser.parse_args()

    target = args.target or detect_host_target()
    if not target:
        print("Warning: could not detect host target; pass --target or place the server sidecar manually.")
        return 0

    is_windows = target.endswith("windows-msvc")
    sidecar_name = f"chirp-server-{target}" + (".exe" if is_windows else "")
    dest_dir = ROOT / "chirp-desktop" / "src-tauri" / "binaries"
    dest = dest_dir / sidecar_name
    profile_args = [] if args.profile == "debug" else ["--release"]
    cmd = [
        "cargo",
        "build",
        "-p",
        "chirp-server",
        "--bin",
        "chirp-server",
        "--target",
        target,
        *profile_args,
    ]
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)

    source = ROOT / "target" / target / args.profile / ("chirp-server.exe" if is_windows else "chirp-server")
    if not source.exists():
        raise FileNotFoundError(source)
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)
    if not is_windows:
        dest.chmod(dest.stat().st_mode | 0o111)
    print(f"Installed Chirp server sidecar at {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
