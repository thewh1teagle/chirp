#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import shutil
import tarfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def host_name() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()
    machine = {"amd64": "x64", "x86_64": "x64", "aarch64": "arm64"}.get(machine, machine)
    return f"{system}-{machine}"


def copy_matches(src: Path, dst: Path, patterns: list[str]) -> list[str]:
    copied: list[str] = []
    for pattern in patterns:
        for path in sorted(src.glob(pattern)):
            if path.is_file() or path.is_symlink():
                out = dst / path.name
                shutil.copy2(path, out, follow_symlinks=True)
                copied.append(out.name)
    return copied


def make_archive(staging: Path, archive: Path) -> None:
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in staging.rglob("*"):
                if path.is_file():
                    zf.write(path, path.relative_to(staging.parent))
        return
    with tarfile.open(archive, "w:gz") as tf:
        tf.add(staging, arcname=staging.name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Package chirp-c native libraries")
    parser.add_argument("--build-dir", type=Path, default=ROOT / "chirp-c" / "build")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "dist")
    parser.add_argument("--backend", default="cpu")
    parser.add_argument("--platform", default=host_name())
    parser.add_argument("--archive", action="store_true", help="create .tar.gz/.zip archive")
    args = parser.parse_args()

    name = f"chirp-c-{args.platform}-{args.backend}"
    staging = args.out_dir / name
    include_dir = staging / "include"
    lib_dir = staging / "lib"
    if staging.exists():
        shutil.rmtree(staging)
    include_dir.mkdir(parents=True)
    lib_dir.mkdir(parents=True)

    shutil.copy2(ROOT / "chirp-c" / "src" / "qwen3_tts.h", include_dir / "qwen3_tts.h")

    copied: list[str] = []
    copied += copy_matches(args.build_dir, lib_dir, ["libchirp-runtime-lib.a", "chirp-runtime-lib.lib"])
    copied += copy_matches(args.build_dir / "_deps" / "llama_cpp-build" / "ggml" / "src", lib_dir, [
        "libggml*.so*",
        "libggml*.dylib*",
        "libggml*.a",
        "ggml*.dll",
        "ggml*.lib",
    ])
    copied += copy_matches(args.build_dir / "_deps" / "kissfft-build", lib_dir, [
        "libkissfft*.a",
        "libkissfft*.so*",
        "libkissfft*.dylib*",
        "kissfft*.lib",
        "kissfft*.dll",
    ])
    copied += copy_matches(args.build_dir / "_deps" / "soxr-build" / "src", lib_dir, [
        "libsoxr*.a",
        "libsoxr*.so*",
        "libsoxr*.dylib*",
        "soxr*.lib",
        "soxr*.dll",
    ])

    metadata = {
        "name": name,
        "backend": args.backend,
        "platform": args.platform,
        "include": ["qwen3_tts.h"],
        "libs": sorted(set(copied)),
    }
    (staging / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    if args.archive:
        archive = args.out_dir / (name + (".zip" if platform.system().lower() == "windows" else ".tar.gz"))
        if archive.exists():
            archive.unlink()
        make_archive(staging, archive)
        print(archive)
    else:
        print(staging)


if __name__ == "__main__":
    main()
