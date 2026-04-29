<p align="center">
  <a target="_blank" href="https://thewh1teagle.github.io/chirp/">
    <img width="96" alt="Chirp logo" src="./chirp-desktop/src/assets/chirp-logo.svg" />
  </a>
</p>

<h1 align="center">Chirp</h1>

<p align="center">
  <strong>Native offline Qwen3-TTS for desktop</strong>
  <br />
</p>

<p align="center">
  <a target="_blank" href="https://thewh1teagle.github.io/chirp/">Download Chirp</a>
  &nbsp; | &nbsp;
  <a target="_blank" href="https://github.com/thewh1teagle/chirp">Give it a Star ⭐</a>
  &nbsp; | &nbsp;
  <a target="_blank" href="https://github.com/sponsors/thewh1teagle">Support the project 🤝</a>
</p>

<hr />

<p align="center">
  <a target="_blank" href="https://thewh1teagle.github.io/chirp/">
    <img width="800" alt="Chirp desktop screenshot" src="https://github.com/user-attachments/assets/6f8fca35-bf3b-4bdb-bdcc-1c52f535e04d" />
  </a>
</p>

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
