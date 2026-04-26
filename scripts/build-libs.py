#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import platform
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def default_preset() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower().replace("amd64", "x86_64")
    return f"{system}-{machine}-cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build chirp-c native libraries")
    parser.add_argument("--backend", default="cpu", choices=["cpu"], help="backend to build")
    parser.add_argument("--build-dir", type=Path, default=ROOT / "chirp-c" / "build")
    parser.add_argument("--config", default="Release")
    parser.add_argument("--target", default="chirp-runtime-lib")
    parser.add_argument("--generator", default=os.environ.get("CMAKE_GENERATOR", ""))
    parser.add_argument("--preset-name", default=default_preset())
    args = parser.parse_args()

    src_dir = ROOT / "chirp-c"
    args.build_dir.mkdir(parents=True, exist_ok=True)

    configure = [
        "cmake",
        "-S",
        str(src_dir),
        "-B",
        str(args.build_dir),
        "-DCMAKE_BUILD_TYPE=" + args.config,
    ]
    if args.generator:
        configure.extend(["-G", args.generator])
    run(configure)
    run(["cmake", "--build", str(args.build_dir), "--target", args.target, "--config", args.config, "-j"])
    print(f"built {args.target} in {args.build_dir} ({args.preset_name})")


if __name__ == "__main__":
    main()
