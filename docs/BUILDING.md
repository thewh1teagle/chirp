# Building

Chirp is built from the Rust workspace and the Tauri desktop app. The desktop bundles a Rust sidecar named `chirp-server`.

## Server

Build the local HTTP server:

```console
cargo build -p chirp-server --release --bin chirp-server
```

For a specific Tauri sidecar target:

```console
uv run scripts/pre_build.py --target aarch64-apple-darwin
```

That command builds `chirp-server` with Cargo and copies the platform-suffixed binary into:

```text
chirp-desktop/src-tauri/binaries/
```

## Desktop

Install frontend dependencies:

```console
cd chirp-desktop
pnpm install
```

Run the app in development:

```console
pnpm tauri dev
```

Build a package:

```console
pnpm tauri build
```

## Models

Packaged model releases use `chirp-models-v*` tags. The Qwen bundle layout is:

```text
chirp-models-q5_0/
  qwen3-tts-model.gguf
  qwen3-tts-codec.gguf
  metadata.json
```

Kokoro bundles contain:

```text
chirp-kokoro-models-kokoro-v1.0/
  kokoro-v1.0.onnx
  voices-v1.0.bin
  espeak-ng-data/
  manifest.json
```

## Releases

Server sidecar releases use `chirp-server-v*` tags:

```console
git tag chirp-server-v0.1.0
git push origin chirp-server-v0.1.0
```

Manual release workflow:

```console
gh workflow run release-chirp-server.yml \
  --ref main \
  -f version=chirp-server-v0.1.0
```

## Checks

```console
cargo test --workspace
cargo build -p chirp-server --release --bin chirp-server
cd chirp-desktop
pnpm build
pnpm tauri build --debug
```
