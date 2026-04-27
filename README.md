# Chirp

Native Qwen3-TTS inference experiments.

## Features

- Local text-to-speech with Qwen3-TTS
- Fully offline generation after the model is downloaded
- Voice cloning from a reference WAV
- Create speech in 10 supported languages
- Audio preview after creation
- CLI support for creating WAV files
- Local HTTP API for app and automation use

## Layout

```console
chirp-c/   C/C++ runtime, C API, GGUF conversion scripts
chirp-runner/ Go CLI and HTTP server using chirp-c through cgo
chirp-desktop/ Tauri + React desktop app
plans/     implementation notes and validation logs
```

## Build

See [BUILDING.md](docs/BUILDING.md).
