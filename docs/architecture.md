# Architecture

Chirp is split into three layers: a native inference runtime, a Go runner, and a desktop app.

## Native Runtime

`runtimes/qwen/` owns model execution. It is a C/C++ library with a small C API in `runtimes/qwen/src/qwen3_tts.h`.

Responsibilities:

- Load Qwen3-TTS GGUF model weights.
- Load the GGUF codec/tokenizer model.
- Run text tokenization, speaker embedding extraction, AR code generation, codec decoding, and WAV writing.
- Use ggml backends for acceleration: Metal on macOS and Vulkan on Linux/Windows release builds.

The C API is the stable boundary. Higher layers should call `qwen3_tts_init`, `qwen3_tts_synthesize_to_file`, and `qwen3_tts_free` instead of reaching into C++ internals.

Release artifacts use `chirp-c-v*` tags for compatibility:

- `chirp-c-darwin-arm64-metal.tar.gz`
- `chirp-c-linux-x64-vulkan.tar.gz`
- `chirp-c-linux-arm64-vulkan.tar.gz`
- `chirp-c-windows-x64-vulkan.zip`

Windows native libraries are built with MSYS2 UCRT64/MinGW so they link cleanly with Go cgo. The package normalizes MinGW archive names to `lib*.a`, matching `-l...` linker flags.

## Models

Model files are stored outside git under `models/`.

Packaged model releases use `chirp-models-v*` tags. The current bundle layout is:

```text
chirp-models-q5_0/
  qwen3-tts-model.gguf
  qwen3-tts-codec.gguf
  metadata.json
```

The model bundle contains the AR model and codec GGUF. It does not contain the native runtime or runner binary.

## Runner

`chirp-runner/` is a Go CLI and HTTP server around the C API.

Responsibilities:

- Provide `chirp speak` for command-line synthesis.
- Provide `chirp serve` for local HTTP usage.
- Link against released or locally built `runtimes/qwen` libraries with cgo.
- Keep process/server concerns out of the native runtime.

During development, the runner can link to `runtimes/qwen/build`. Release builds download a pinned Qwen native archive into `chirp-runner/third_party/chirp-c` first.

Runner release artifacts use `chirp-runner-v*` tags:

- `chirp-runner-darwin-arm64.tar.gz`
- `chirp-runner-linux-x64.tar.gz`
- `chirp-runner-linux-arm64.tar.gz`
- `chirp-runner-windows-x64.zip`

The runner is intentionally separate from the model bundle. Users can update native code, runner code, and models independently.

## Desktop

`chirp-desktop/` is the Tauri + React app.

Responsibilities:

- Own the user interface.
- Use the runner as the local inference service boundary.
- Avoid linking directly to ggml or the C++ runtime.

The desktop app should treat the runner as an external local service or subprocess. That keeps UI packaging separate from native inference releases and lets the desktop app reuse prebuilt runner binaries.

## Release Flow

Typical release order:

1. Build and release native libraries: `chirp-c-v*`.
2. Build and release runner binaries pinned to a native release: `chirp-runner-v*`.
3. Publish model bundles when model files change: `chirp-models-v*`.
4. Build desktop packages against a selected runner release.

`test-release.yml` downloads released runner and model artifacts and runs a small synthesis test on each platform.
