# Chirp

Native Qwen3-TTS inference experiments.

## Layout

```text
chirp-c/   C/C++ runtime, C API, GGUF conversion scripts
chirp-runner/ Go CLI and HTTP server using chirp-c through cgo
chirp-desktop/ Tauri + React desktop app
plans/     implementation notes and validation logs
```

## Build

See [BUILDING.md](docs/BUILDING.md).
