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
    return f"{system}-{machine}-{default_backend()}"


def default_backend() -> str:
    system = platform.system()
    if system == "Darwin":
        return "metal"
    if system in ("Linux", "Windows"):
        return "vulkan"
    return "cpu"


def default_generator() -> str:
    generator = os.environ.get("CMAKE_GENERATOR", "")
    if generator:
        return generator
    if platform.system() == "Windows":
        return "MinGW Makefiles"
    return ""


def build_vulkan_delay_lib(build_dir: Path) -> None:
    if platform.system() != "Windows":
        return
    delay_dir = build_dir / "delay"
    delay_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(["where", "vulkan-1.dll"], capture_output=True, text=True)
    dll_path = "vulkan-1.dll"
    if result.returncode == 0 and result.stdout.strip():
        dll_path = result.stdout.strip().splitlines()[0]
    run(["gendef", dll_path], cwd=delay_dir)
    run(["dlltool", "--input-def", "vulkan-1.def", "--output-delaylib", "libvulkan-1-delay.a"], cwd=delay_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build runtimes/qwen native libraries")
    parser.add_argument("--backend", default=default_backend(), choices=["cpu", "metal", "vulkan"], help="backend to build")
    parser.add_argument("--build-dir", type=Path, default=ROOT / "runtimes/qwen" / "build")
    parser.add_argument("--config", default="Release")
    parser.add_argument("--target", default="chirp-runtime-lib")
    parser.add_argument("--generator", default=default_generator())
    parser.add_argument("--preset-name", default=default_preset())
    args = parser.parse_args()

    src_dir = ROOT / "runtimes/qwen"
    args.build_dir.mkdir(parents=True, exist_ok=True)

    configure = [
        "cmake",
        "-S",
        str(src_dir),
        "-B",
        str(args.build_dir),
        "-DCMAKE_BUILD_TYPE=" + args.config,
        "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
    ]
    if args.backend == "metal":
        configure.extend(["-DGGML_METAL=ON", "-DGGML_METAL_EMBED_LIBRARY=ON"])
    elif args.backend == "vulkan":
        configure.append("-DGGML_VULKAN=ON")
    if args.generator:
        configure.extend(["-G", args.generator])
    run(configure)
    run(["cmake", "--build", str(args.build_dir), "--target", args.target, "--config", args.config, "-j"])
    if args.backend == "vulkan":
        build_vulkan_delay_lib(args.build_dir)
    print(f"built {args.target} in {args.build_dir} ({args.preset_name})")


if __name__ == "__main__":
    main()
