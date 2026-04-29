#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def host_name() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()
    machine = {"amd64": "x64", "x86_64": "x64", "aarch64": "arm64"}.get(machine, machine)
    return f"{system}-{machine}"


def github_json(url: str) -> dict:
    headers = {"Accept": "application/vnd.github+json"}
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as res:
        return json.loads(res.read().decode("utf-8"))


def download(url: str, dest: Path, token: str | None = None) -> None:
    print(f"downloading {url}")
    headers = {}
    if token:
        headers["Accept"] = "application/octet-stream"
        headers["Authorization"] = f"Bearer {token}"
        headers["X-GitHub-Api-Version"] = "2022-11-28"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as res, dest.open("wb") as out:
        shutil.copyfileobj(res, out)


def extract(archive: Path, dest: Path) -> None:
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
        return
    with tarfile.open(archive, "r:gz") as tf:
        tf.extractall(dest)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download prebuilt Qwen native libraries from GitHub releases")
    parser.add_argument("--repo", default="thewh1teagle/chirp", help="GitHub repo owner/name")
    parser.add_argument("--version", default="latest", help="release tag or latest")
    parser.add_argument("--tag", help="release tag; overrides --version")
    parser.add_argument("--backend", default="cpu")
    parser.add_argument("--platform", default=host_name())
    parser.add_argument("--out-dir", type=Path, default=ROOT / "chirp-runner" / "third_party" / "chirp-c")
    args = parser.parse_args()

    stem = f"chirp-c-{args.platform}-{args.backend}"
    version = args.tag or args.version
    api = f"https://api.github.com/repos/{args.repo}/releases/latest" if version == "latest" else f"https://api.github.com/repos/{args.repo}/releases/tags/{version}"
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    release = github_json(api)
    assets = release.get("assets", [])
    asset = next((a for a in assets if a.get("name", "").startswith(stem + ".tar.gz") or a.get("name", "").startswith(stem + ".zip")), None)
    if not asset:
        names = ", ".join(a.get("name", "") for a in assets)
        raise SystemExit(f"asset not found for {stem}; available: {names}")

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        archive = tmp / asset["name"]
        url = asset["url"] if token else asset["browser_download_url"]
        download(url, archive, token)
        extract(archive, tmp)
        extracted = tmp / stem
        if not extracted.exists():
            raise SystemExit(f"archive did not contain {stem}")
        if args.out_dir.exists():
            shutil.rmtree(args.out_dir)
        args.out_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(extracted, args.out_dir)
    print(args.out_dir)


if __name__ == "__main__":
    main()
